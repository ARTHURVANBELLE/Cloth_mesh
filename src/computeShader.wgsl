struct Instance {
    position: vec4<f32>, // Position as vec4
    velocity: vec4<f32>, // Velocity as vec4
};

@group(0) @binding(0) var<storage, read_write> fabric_vertices: array<Instance>;

const gravity: vec4<f32> = vec4<f32>(0.0, -0.81, 0.0, 0.0); // Gravity as vec4
const delta_time: f32 = 0.016;
const floor_height: f32 = -1.0; // Define the floor height

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;

    // Load vertex data
    var vertex = fabric_vertices[index];

    // Apply gravity: assign to velocity components individually
    vertex.velocity.x += gravity.x * delta_time;
    vertex.velocity.y += gravity.y * delta_time;
    vertex.velocity.z += gravity.z * delta_time;

    // Update position: assign to position components individually
    vertex.position.x += vertex.velocity.x * delta_time;
    vertex.position.y += vertex.velocity.y * delta_time;
    vertex.position.z += vertex.velocity.z * delta_time;

    // Floor collision
    if (vertex.position.y < floor_height) {
        vertex.position.y = floor_height;       // Reset to floor height
    }

    // Store updated vertex data
    fabric_vertices[index] = vertex;
}
