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
use rand::Rng; // Add this line

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
    mass: f32,
    velocity: [f32; 3],
    is_ball: i32,
}

#[derive(Copy, Clone, Debug)]
struct Spring {
    vertex1: usize,
    vertex2: usize,
    rest_length: f32,
    stiffness: f32,
}

pub struct InstanceApp {
    vertex_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    render_pipeline: wgpu::RenderPipeline,
    num_indices: u32,
    num_instances: u32,
    camera: OrbitCamera,
    instances: Vec<Instance>,
    vertices: Vec<Vertex>,
    springs: Vec<Spring>,
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Instance {
    position: [f32; 3],
    velocity: [f32; 3],
}

impl Instance {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Instance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

impl InstanceApp {
    pub fn new(context: &Context) -> Self {
        // Function to create cloth mesh (cloth vertices and springs)
        fn create_cloth_mesh(grid_size: usize, spacing: f32) -> (Vec<Vertex>, Vec<Spring>) {
            let mut vertices = Vec::new();
            let mut springs = Vec::new();

            // Generate cloth vertices and springs
            for i in 0..grid_size {
                for j in 0..grid_size {
                    let position = [
                        i as f32 * spacing - (grid_size as f32 * spacing / 2.0),
                        1.0, // Initial height of the cloth
                        j as f32 * spacing - (grid_size as f32 * spacing / 2.0),
                    ];

                    // Add cloth vertex (with blue color)
                    vertices.push(Vertex {
                        position,
                        color: [0.0, 0.0, 1.0], // Blue for the cloth
                        mass: 0.1,
                        velocity: [0.0, 0.0, 0.0],
                        is_ball: 0, // Not a ball
                    });

                    // Add springs connecting adjacent vertices
                    if i > 0 {
                        springs.push(Spring {
                            vertex1: i * grid_size + j,
                            vertex2: (i - 1) * grid_size + j,
                            rest_length: spacing,
                            stiffness: 50.0,
                        });
                    }
                    if j > 0 {
                        springs.push(Spring {
                            vertex1: i * grid_size + j,
                            vertex2: i * grid_size + (j - 1),
                            rest_length: spacing,
                            stiffness: 50.0,
                        });
                    }

                    // Shear springs
                    if i > 0 && j > 0 {
                        springs.push(Spring {
                            vertex1: i * grid_size + j,
                            vertex2: (i - 1) * grid_size + (j - 1),
                            rest_length: (2.0 * spacing * spacing).sqrt(),
                            stiffness: 50.0,
                        });
                        springs.push(Spring {
                            vertex1: i * grid_size + (j - 1),
                            vertex2: (i - 1) * grid_size + j,
                            rest_length: (2.0 * spacing * spacing).sqrt(),
                            stiffness: 50.0,
                        });
                    }

                    // Bend springs
                    if i > 1 {
                        springs.push(Spring {
                            vertex1: i * grid_size + j,
                            vertex2: (i - 2) * grid_size + j,
                            rest_length: 2.0 * spacing,
                            stiffness: 25.0,
                        });
                    }
                    if j > 1 {
                        springs.push(Spring {
                            vertex1: i * grid_size + j,
                            vertex2: i * grid_size + (j - 2),
                            rest_length: 2.0 * spacing,
                            stiffness: 25.0,
                        });
                    }
                }
            }

            (vertices, springs)
        }

        // Create ball and cloth mesh
        let grid_size = 10;
        let spacing = 0.1;
        let (cloth_vertices, cloth_springs) = create_cloth_mesh(grid_size, spacing);

        // Generate the ball shape (icosphere)
        let (ball_positions, ball_indices) = icosphere(2);
        let mut rng = rand::thread_rng();

        // Create ball vertices (red color)
        let ball_vertices: Vec<Vertex> = ball_positions
            .iter()
            .map(|position| Vertex {
                position: (*position * 0.2).into(),
                color: [1.0, 0.0, 0.0], // Red for the ball
                mass: 0.1,
                velocity: [0.0, 0.0, 0.0],
                is_ball: 1, // It's a ball
            })
            .collect();

        // Combine ball and cloth vertices
        let mut vertices = ball_vertices;
        vertices.extend(cloth_vertices);

        // Combine ball and cloth springs (only springs for the cloth mesh in this case)
        let mut springs = cloth_springs;
        
        // Generate cloth indices (for rendering)
        let cloth_indices = generate_cloth_indices(grid_size);

        // Create instances for ball positions
        let ball_instances: Vec<Instance> = ball_positions
            .iter()
            .map(|position| Instance {
                position: (*position).into(),
                velocity: [
                    rng.gen_range(-0.1..0.1),
                    rng.gen_range(-0.1..0.1),
                    rng.gen_range(-0.1..0.1),
                ],
            })
            .collect();

        // Buffers for vertex, instance, and index data
        let index_buffer = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(cloth_indices.as_slice()),
                usage: wgpu::BufferUsages::INDEX,
            });

        let vertex_buffer = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(vertices.as_slice()),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let instance_buffer = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(ball_instances.as_slice()),
                usage: wgpu::BufferUsages::VERTEX,
            });
        
        let shader = context
        .device()
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let camera_bind_group_layout = context
            .device()
            .create_bind_group_layout(&CameraUniform::desc());

        let pipeline_layout =
        context
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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
                        // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                        polygon_mode: wgpu::PolygonMode::Fill,
                        // Requires Features::DEPTH_CLIP_CONTROL
                        unclipped_depth: false,
                        // Requires Features::CONSERVATIVE_RASTERIZATION
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
        
        let aspect = context.size().x / context.size().y;
        // Return instance app
        InstanceApp {
            vertex_buffer,
            instance_buffer,
            index_buffer,
            render_pipeline,
            num_indices: cloth_indices.len() as u32,
            num_instances: ball_instances.len() as u32,
            camera: OrbitCamera::new(context, 45.0, aspect, 0.1, 100.0),
            instances: ball_instances,
            vertices,
            springs,
        }
    }
}


fn generate_cloth_indices(grid_size: usize) -> Vec<u32> {
    let mut indices = Vec::new();

    // Loop over grid to create indices for the cloth mesh
    for i in 0..grid_size - 1 {
        for j in 0..grid_size - 1 {
            let current = i * grid_size + j;
            let right = current + 1;
            let down = current + grid_size;
            let down_right = down + 1;

            // Create two triangles per square in the grid
            indices.push(current as u32);
            indices.push(down as u32);
            indices.push(right as u32);

            indices.push(right as u32);
            indices.push(down as u32);
            indices.push(down_right as u32);
        }
    }

    indices
}

impl App for InstanceApp {
    fn input(&mut self, input: egui::InputState, context: &Context) {
        self.camera.input(input.clone(), context);

        let input_raw_scroll = input.raw_scroll_delta.y.clone();
        if input_raw_scroll != 0.0 {
            let new_radius = (self.camera.radius() - input.raw_scroll_delta.y).max(0.1).min(500.0); // Prevent negative or zero radius
            self.camera.set_radius(new_radius).update(context);
        }
    }

    fn render(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.set_bind_group(0, self.camera.bind_group(), &[]);
        render_pass.draw_indexed(0..self.num_indices, 0, 0..self.num_instances);
        
    }
}