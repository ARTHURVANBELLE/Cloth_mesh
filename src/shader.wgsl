// Camera uniform struct
struct CameraUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

// Vertex structure input
struct VertexInput {
    @location(0) position: vec3<f32>, // Vertex position (from Vertex)
    @location(1) color: vec3<f32>,    // Vertex color (from Vertex)
    @location(2) mass: f32,           // Vertex mass (from Vertex)
    @location(3) velocity: vec3<f32>, // Vertex velocity (from Vertex)
    @location(4) is_ball: f32,       // Vertex is_ball (from Vertex)
};

// Instance structure input
struct InstanceInput {
    @location(5) position: vec3<f32>, // Instance position
    @location(6) velocity: vec3<f32>, // Instance velocity
};

// Vertex structure output
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>, // Output color to fragment shader
};

@vertex
fn vs_main(
    model: VertexInput,        // Vertex input data
    instance: InstanceInput,   // Instance input data
) -> VertexOutput {
    var out: VertexOutput;
    
    // Color selection based on `is_ball` (1 for ball, else default)
    out.color = select(model.color, vec3<f32>(1.0, 0.0, 0.0), model.is_ball == 1.0);
    
    // Compute the clip position (model + instance position, with camera projection)
    out.clip_position = camera.proj * camera.view * vec4<f32>(model.position + instance.position, 1.0);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0); // Full opacity color
}
