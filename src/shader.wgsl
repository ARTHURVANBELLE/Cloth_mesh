// Camera uniform struct
struct CameraUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

// Vertex structure input
struct VertexInput {
    @location(0) position: vec4<f32>, // Position as vec4
    @location(1) color: vec4<f32>,   // Use vec4 instead of vec3
    @location(2) mass: f32,          // Single scalar value
    @location(3) velocity: vec4<f32>, // Velocity as vec4
    @location(4) is_ball: f32,       // Single scalar value
};

// Vertex structure output
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>, // Output clip position
    @location(0) color: vec4<f32>,              // Output color as vec4
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Use the scalar `is_ball` to determine logic
    let is_ball = model.is_ball;

    // Use vec4 for color. Alpha is defaulted to 1.0 if it's a ball.
    out.color = select(model.color, vec4<f32>(1.0, 0.0, 0.0, 0.0), is_ball == 1.0);

    // Calculate the clip position using the vec4 position
    out.clip_position = camera.proj * camera.view * model.position;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Return color as vec4 with full opacity
    return in.color;
}
