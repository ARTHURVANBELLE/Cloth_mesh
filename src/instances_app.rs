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
}

#[derive(Copy, Clone, Debug)]
struct Spring {
    vertex1: usize,
    vertex2: usize,
    rest_length: f32,
    stiffness: f32,
}

pub struct Particle {
    position: [f32; 3],
    velocity: [f32; 3],
    color: [f32; 3],
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
    particles: Vec<Particle>,
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
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                shader_location: 3,
                format: wgpu::VertexFormat::Float32x3,
            },],
            
        }
    }
}

impl InstanceApp {
    pub fn new(context: &Context) -> Self {
        let (positions, indices) = icosphere(2);
        let mut rng = rand::thread_rng();

        let grid_size = 10;
        let spacing = 0.1;

        let vertices: Vec<Vertex> = positions
            .iter()
            .map(|position| Vertex {
                position: (*position * 0.2).into(),
                color: [1.0, 0.0, 0.0],
                mass: 0.1,
                velocity: [0.0, 0.0, 0.0],
            })
            .collect();

        let mut springs: Vec<Spring> = (0..grid_size)
            .flat_map(|i| {
                (0..grid_size).map(move |j| {
                    let vertex1 = i * grid_size + j;
                    let vertex2 = i * grid_size + j + 1;
                    Spring {
                        vertex1,
                        vertex2,
                        rest_length: spacing,
                        stiffness: 0.1,
                    }
                })
            })
            .chain((0..grid_size).flat_map(|j| {
                (0..grid_size).map(move |i| {
                    let vertex1 = i * grid_size + j;
                    let vertex2 = (i + 1) * grid_size + j;
                    Spring {
                        vertex1,
                        vertex2,
                        rest_length: spacing,
                        stiffness: 0.1,
                    }
                })
            }))
            .collect();

        let particles: Vec<Particle> = (0..6) // Generate 100 particles
            .map(|_| Particle {
                position: [rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0)],
                velocity: [rng.gen_range(-0.1..0.1), rng.gen_range(-0.1..0.1), rng.gen_range(-0.1..0.1)],
                color: [rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0)],
            })
            .collect();

        let index_buffer = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(indices.as_slice()),
                usage: wgpu::BufferUsages::INDEX,
            });

        let mut rng = rand::thread_rng(); // Add this line
        let instances: Vec<Instance> = positions
            .iter()
            .map(|position| Instance {
                position: (*position).into(),
                velocity: [
                    rng.gen_range(-0.1..0.1), // Modify this line
                    rng.gen_range(-0.1..0.1), // Modify this line
                    rng.gen_range(-0.1..0.1), // Modify this line
                ],
            })
            .collect();

        //let num_indices = indices.len() as u32;
        //let num_instances = instances.len() as u32;

        let vertex_buffer =
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer"),
                    contents: bytemuck::cast_slice(vertices.as_slice()),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        let instance_buffer =
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Instance Buffer"),
                    contents: bytemuck::cast_slice(instances.as_slice()),
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
        //let camera = OrbitCamera::new(context, 45.0, aspect, 0.1, 100.0);

        // Create structural springs
        for i in 0..grid_size {
            for j in 0..grid_size {
                if i < grid_size - 1 {
                    springs.push(Spring {
                        vertex1: i * grid_size + j,
                        vertex2: (i + 1) * grid_size + j,
                        rest_length: spacing,
                        stiffness: 100.0,
                    });
                }
                if j < grid_size - 1 {
                    springs.push(Spring {
                        vertex1: i * grid_size + j,
                        vertex2: i * grid_size + (j + 1),
                        rest_length: spacing,
                        stiffness: 100.0,
                    });
                }
            }
        }
        // Create shear springs
        for i in 0..grid_size - 1 {
            for j in 0..grid_size - 1 {
                springs.push(Spring {
                    vertex1: i * grid_size + j,
                    vertex2: (i + 1) * grid_size + (j + 1),
                    rest_length: (2.0 * spacing * spacing).sqrt(),
                    stiffness: 100.0,
                });
                springs.push(Spring {
                    vertex1: (i + 1) * grid_size + j,
                    vertex2: i * grid_size + (j + 1),
                    rest_length: (2.0 * spacing * spacing).sqrt(),
                    stiffness: 100.0,
                });
            }
        }
        // Create bend springs
        for i in 0..grid_size {
            for j in 0..grid_size {
                if i < grid_size - 2 {
                    springs.push(Spring {
                        vertex1: i * grid_size + j,
                        vertex2: (i + 2) * grid_size + j,
                        rest_length: 2.0 * spacing,
                        stiffness: 50.0,
                    });
                }
                if j < grid_size - 2 {
                    springs.push(Spring {
                        vertex1: i * grid_size + j,
                        vertex2: i * grid_size + (j + 2),
                        rest_length: 2.0 * spacing,
                        stiffness: 50.0,
                    });
                }
            }
        }

        Self {
            vertex_buffer,
            instance_buffer,
            index_buffer,
            render_pipeline,
            num_indices: indices.len() as u32,
            num_instances: 1, // Set the number of instances
            camera: OrbitCamera::new(context, 45.0, aspect, 0.4, 100.0),
            instances: vec![], // Initialize instances
            particles, // Initialize particles
            vertices,
            springs,
            }
    }
    fn update_instances(&mut self, context: &Context) {

        for instance in &mut self.instances {
            for i in 0..3 {
                instance.position[i] += instance.velocity[i];
            }
        }

        self.instance_buffer = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Instance Buffer"),
                contents: bytemuck::cast_slice(self.instances.as_slice()),
                usage: wgpu::BufferUsages::VERTEX,
            });
    }

    pub fn update(mut self, context: &Context) {
        self.camera.update(context);
        self.update_instances(context); // Add this line
    }

    fn calculate_forces(&mut self) {
        let gravity = [0.0, -9.81, 0.0];
        let damping = 0.99;

        let positions: Vec<[f32; 3]> = self.vertices.iter().map(|v| v.position).collect();

        for vertex in &mut self.vertices {
            let mut force = [0.0, 0.0, 0.0];

            // Apply gravity
            for i in 0..3 {
                force[i] += vertex.mass * gravity[i];
            }

            // Apply spring forces
            for spring in &self.springs {
                let v1 = positions[spring.vertex1];
                let v2 = positions[spring.vertex2];
                let delta = [
                    v2[0] - v1[0],
                    v2[1] - v1[1],
                    v2[2] - v1[2],
                ];
                let distance = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
                let force_magnitude = spring.stiffness * (distance - spring.rest_length);
                for i in 0..3 {
                    force[i] += force_magnitude * delta[i] / distance;
                }
            }

            // Apply damping
            for i in 0..3 {
                vertex.velocity[i] *= damping;
            }

            // Update velocity and position
            for i in 0..3 {
                vertex.velocity[i] += force[i] / vertex.mass;
                vertex.position[i] += vertex.velocity[i];
            }
        }
    }
}

impl App for InstanceApp {
    fn input(&mut self, input: egui::InputState, context: &Context) {
        self.camera.input(input.clone(), context);

        let input_raw_scroll = input.raw_scroll_delta.y.clone();
        if input_raw_scroll != 0.0 {
            let new_radius = (self.camera.radius() - input.raw_scroll_delta.y).max(0.1).min(5.0); // Prevent negative or zero radius
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
