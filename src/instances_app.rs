use std::cmp::max;

use wgpu_bootstrap::{
    cgmath, egui,
    util::{
        geometry::icosphere,
        orbit_camera::{CameraUniform, OrbitCamera},
    },
    wgpu::{self, util::DeviceExt},
    App, Context,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
    mass: f32,
    velocity: [f32; 3],
    is_ball: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Instance {
    position: [f32; 3],
    velocity: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Spring {
    vertex_a: usize,
    vertex_b: usize,
    rest_length: f32,
    stiffness: f32,
}


impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Location 0: Position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Location 1: Color
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Location 2: Mass
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32,
                },
                // Location 3: Velocity
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() + std::mem::size_of::<f32>()) as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Location 4: is_ball
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() * 2 + std::mem::size_of::<f32>()) as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}


impl Instance {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Instance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

pub struct InstanceApp {
    sphere_vertex_buffer: wgpu::Buffer,
    sphere_index_buffer: wgpu::Buffer,
    fabric_vertex_buffer: wgpu::Buffer,
    fabric_index_buffer: wgpu::Buffer,
    render_pipeline: wgpu::RenderPipeline,
    num_sphere_indices: u32,
    num_fabric_indices: u32,
    num_instances: u32,
    camera: OrbitCamera,
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
}

impl InstanceApp {
    pub fn new(context: &Context) -> Self {
        

        let rows = 20;
        let cols = 20;
        let spacing = 0.5;
        let stiffness = 100.0;

        let ball_radius = 0.2; // Adjust the ball radius as needed
        let (ball_positions, ball_indices) = icosphere(2);
        let ball_vertices: Vec<Vertex> = ball_positions
            .iter()
            .map(|position| Vertex {
                position: (*position * ball_radius).into(),
                color: [1.0, 0.0, 0.0], // Red for the ball
                mass: 1.0,
                velocity: [0.0, 0.0, 0.0],
                is_ball: 1.0,
            })
            .collect();
        
        let fabric_vertices = create_fabric_vertices(rows, cols, spacing, ball_radius);


        fn create_fabric_vertices(rows: usize, cols: usize, spacing: f32, ball_radius: f32) -> Vec<Vertex> {
            let mut vertices = Vec::new();
            for i in 0..rows {
                for j in 0..cols {
                    let x = -ball_radius + j as f32 * spacing;
                    let z = -ball_radius + i as f32 * spacing;
                    let y = ball_radius + 0.1; // Slightly above the ball
                    vertices.push(Vertex {
                        position: [x, y, z],
                        color: [0.0, 1.0, 0.0], // Green color
                        mass: 1.0,
                        velocity: [0.0, 0.0, 0.0],
                        is_ball: 0.0,
                    });
                }
            }
            vertices
        }
        

        fn create_springs(
            rows: usize,
            cols: usize,
            vertices: &Vec<Vertex>,
            stiffness: f32,
        ) -> Vec<Spring> {
            let mut springs = Vec::new();
        
            for row in 0..rows {
                for col in 0..cols {
                    let index = row * cols + col;
        
                    // Structural springs
                    if col < cols - 1 {
                        let right = index + 1;
                        springs.push(Spring {
                            vertex_a: index,
                            vertex_b: right,
                            rest_length: (vertices[index].position[0] - vertices[right].position[0]).abs(),
                            stiffness,
                        });
                    }
                    if row < rows - 1 {
                        let down = index + cols;
                        springs.push(Spring {
                            vertex_a: index,
                            vertex_b: down,
                            rest_length: (vertices[index].position[2] - vertices[down].position[2]).abs(),
                            stiffness,
                        });
                    }
        
                    // Shear springs
                    if col < cols - 1 && row < rows - 1 {
                        let down_right = index + cols + 1;
                        springs.push(Spring {
                            vertex_a: index,
                            vertex_b: down_right,
                            rest_length: ((vertices[index].position[0] - vertices[down_right].position[0]).powi(2)
                                + (vertices[index].position[2] - vertices[down_right].position[2]).powi(2))
                            .sqrt(),
                            stiffness,
                        });
                    }
                    if col > 0 && row < rows - 1 {
                        let down_left = index + cols - 1;
                        springs.push(Spring {
                            vertex_a: index,
                            vertex_b: down_left,
                            rest_length: ((vertices[index].position[0] - vertices[down_left].position[0]).powi(2)
                                + (vertices[index].position[2] - vertices[down_left].position[2]).powi(2))
                            .sqrt(),
                            stiffness,
                        });
                    }
        
                    // Bend springs
                    if col < cols - 2 {
                        let two_right = index + 2;
                        springs.push(Spring {
                            vertex_a: index,
                            vertex_b: two_right,
                            rest_length: (vertices[index].position[0] - vertices[two_right].position[0]).abs(),
                            stiffness,
                        });
                    }
                    if row < rows - 2 {
                        let two_down = index + 2 * cols;
                        springs.push(Spring {
                            vertex_a: index,
                            vertex_b: two_down,
                            rest_length: (vertices[index].position[2] - vertices[two_down].position[2]).abs(),
                            stiffness,
                        });
                    }
                }
            }
        
            springs
        }
        
        fn generate_fabric_indices(width_segments: usize, height_segments: usize) -> Vec<u32> {
            let mut indices = Vec::new();
            for row in 0..height_segments {
                for col in 0..width_segments {
                    let top_left = (row * (width_segments + 1) + col) as u32;
                    let top_right = top_left + 1;
                    let bottom_left = ((row + 1) * (width_segments + 1) + col) as u32;
                    let bottom_right = bottom_left + 1;
        
                    // First triangle (top-left, bottom-left, top-right)
                    indices.push(top_left);
                    indices.push(bottom_left);
                    indices.push(top_right);
        
                    // Second triangle (top-right, bottom-left, bottom-right)
                    indices.push(top_right);
                    indices.push(bottom_left);
                    indices.push(bottom_right);
                }
            }
            indices
        }
        
        

        // Generate fabric vertices and indices
        let fabric_indices = generate_fabric_indices(cols - 1, rows - 1);
        println!("Total fabric vertices: {}", fabric_vertices.len());
        println!("Total fabric indices: {}", fabric_indices.len());

        // Combine vertices and indices for rendering
        let mut vertices = Vec::new();
        vertices.extend(&ball_vertices); // Borrow instead of moving
        vertices.extend(fabric_vertices.clone());
        
        let mut indices = Vec::new();
        indices.extend(ball_indices.clone()); // Clone to avoid move
        indices.extend(fabric_indices.iter().map(|i| *i as u32 + ball_vertices.len() as u32));


        println!("Total vertices: {}", vertices.len());
        println!("Total indices: {}", indices.len());

        // Instances for square (initial position and downward velocity)
        let instances = vec![Instance {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
        }];

        let instance_buffer = context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: &[],
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        
        
        let sphere_vertex_buffer = context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sphere Vertex Buffer"),
            contents: bytemuck::cast_slice(&ball_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        
        let sphere_index_buffer = context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sphere Index Buffer"),
            contents: bytemuck::cast_slice(&ball_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        
        let fabric_vertex_buffer = context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fabric Vertex Buffer"),
            contents: bytemuck::cast_slice(&fabric_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        
        let fabric_index_buffer = context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fabric Index Buffer"),
            contents: bytemuck::cast_slice(&fabric_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        

        // Shaders and pipeline
        let shader = context.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let camera_bind_group_layout = context.device().create_bind_group_layout(&CameraUniform::desc());

        let pipeline_layout = context.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let render_pipeline =
            context
                .device()
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Render Pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: "vs_main",
                        buffers: &[Vertex::desc(), Instance::desc()],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: "fs_main",
                        targets: &[Some(wgpu::ColorTargetState {
                            format: context.format(),
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: Some(wgpu::Face::Back),
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: context.depth_stencil_format(),
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Less,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: 1,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview: None,
                    cache: None,
                });

                
        // Camera setup
        let aspect = context.size().x / context.size().y;
        let camera = OrbitCamera::new(context, 45.0, aspect, 0.1, 10.0);

        let num_sphere_indices = ball_indices.len() as u32;
        let num_fabric_indices = fabric_indices.len() as u32;

        InstanceApp {
            sphere_vertex_buffer,
            sphere_index_buffer,
            fabric_vertex_buffer,
            fabric_index_buffer,
            render_pipeline,
            num_sphere_indices,
            num_fabric_indices,
            num_instances: instances.len() as u32,
            camera,
            instances,
            instance_buffer,
        }
        
    }
}

impl App for InstanceApp {
    fn input(&mut self, input: egui::InputState, context: &Context) {
        self.camera.input(input.clone(), context);

        if input.raw_scroll_delta.y != 0.0 {
            let new_radius = (self.camera.radius() - input.raw_scroll_delta.y).max(0.1).min(500.0);
            self.camera.set_radius(new_radius).update(context);
        }
    }

    fn update(&mut self, delta_time: f32, _context: &wgpu_bootstrap::Context<'_>) {
        for instance in &mut self.instances {
            // Update fabric instances (velocity[1] != 0.0) and leave spheres unaffected

            if instance.velocity[1] != 0.0 {
                // Update position based on velocity
                instance.position[1] += instance.velocity[1] * delta_time;
    
                // Apply gravity to fabric instances
                instance.velocity[1] -= 0.8 * delta_time; // Gravity acceleration
    
                // Prevent falling below the ground (example: y = -1.0)
                if instance.position[1] < -1.0 {
                    instance.position[1] = -1.0;
                    instance.velocity[1] = 0.0; // Stop moving when hitting the ground
                }
            }
        }
        let instance_data: Vec<Instance> = self.instances.iter().cloned().collect();
        self.instance_buffer = _context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
    }
    

    fn render(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, self.camera.bind_group(), &[]);
    
        // Draw the sphere
        //render_pass.set_vertex_buffer(0, self.sphere_vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));


        render_pass.set_index_buffer(self.sphere_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.num_sphere_indices, 0, 0..self.num_instances);
    
        // Draw the fabric
        //render_pass.set_vertex_buffer(0, self.fabric_vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

        render_pass.set_index_buffer(self.fabric_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.num_fabric_indices, 0, 0..self.num_instances);
    }
    
    
}
