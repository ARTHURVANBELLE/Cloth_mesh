struct Instance {
    position: vec4<f32>,
    color: vec4<f32>,
    mass: f32,
    velocity: vec4<f32>,
    is_ball: f32,
};

struct SphereData {
    center: vec4<f32>,
    radius: f32,
    _padding: vec3<f32>,  // 12 bytes padding
}

@group(0) @binding(0) var<storage, read_write> fabric_vertices: array<Instance>;
@group(0) @binding(1) var<uniform> sphere: SphereData;

const gravity: vec4<f32> = vec4<f32>(0.0, -9.81, 0.0, 0.0);
const delta_time: f32 = 0.016;
const floor_height: f32 = 0.1;
const damping: f32 = 0.98;

fn handle_sphere_collision(pos: vec4<f32>, vel: vec4<f32>) -> vec4<f32> {
    let to_center = pos.xyz - sphere.center.xyz;
    let distance = length(to_center);
    
    if (distance < sphere.radius) {
        // Move the point to the surface of the sphere
        let normal = normalize(to_center);
        let correction = normal * sphere.radius;
        return vec4<f32>(sphere.center.xyz + correction, pos.w);
    }
    return pos;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&fabric_vertices)) {
        return;
    }

    var vertex = fabric_vertices[index];
    
    if (vertex.is_ball < 0.5) {  // Process only fabric vertices
        // Apply gravity with damping
        vertex.velocity += gravity * delta_time;
        vertex.velocity *= damping;

        // Update position
        vertex.position += vertex.velocity * delta_time;

        // Handle sphere collision
        vertex.position = handle_sphere_collision(vertex.position, vertex.velocity);

        // Floor collision with bounce
        if (vertex.position.y < floor_height) {
            vertex.position.y = floor_height;
            vertex.velocity.y = -vertex.velocity.y * 0.5;
        }
    }

    fabric_vertices[index] = vertex;
}