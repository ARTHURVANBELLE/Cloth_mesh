//compute shader
struct Instance {
    position: vec3<f32>,
    velocity: vec3<f32>,
};

@group(0) @binding(0) var<storage, read_write> fabric_vertices: array<Instance>;

const gravity: vec3<f32> = vec3<f32>(0.0, -9.81, 0.0);
const delta_time: f32 = 0.016;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    let index = id.x;

    var vertex = fabric_vertices[index];

    vertex.velocity = vertex.velocity + gravity * delta_time;

    vertex.position = vertex.position + vertex.velocity * delta_time;

    fabric_vertices[index] = vertex;
}
