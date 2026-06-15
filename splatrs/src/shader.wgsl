struct Uniforms {
    view_proj: mat4x4<f32>,
    viewport: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    options: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexIn {
    @builtin(vertex_index) vertex_index: u32,
    @location(0) position_opacity: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) axis0_radius: vec4<f32>,
    @location(3) axis1_radius: vec4<f32>,
    @location(4) axis2_radius: vec4<f32>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) color: vec4<f32>,
};

fn quad_corner(index: u32) -> vec2<f32> {
    switch index {
        case 0u: { return vec2<f32>(-1.0, -1.0); }
        case 1u: { return vec2<f32>( 1.0, -1.0); }
        case 2u: { return vec2<f32>(-1.0,  1.0); }
        default: { return vec2<f32>( 1.0,  1.0); }
    }
}

@vertex
fn vs_main(input: VertexIn) -> VertexOut {
    let corner = quad_corner(input.vertex_index);
    let center = input.position_opacity.xyz;
    let opacity = input.position_opacity.w * uniforms.options.x;
    let point_mode = uniforms.options.y > 0.5;
    let splat_scale = uniforms.options.z;

    let radius0 = input.axis0_radius.w;
    let radius1 = input.axis1_radius.w;
    let radius2 = input.axis2_radius.w;
    let max_radius = max(radius0, max(radius1, radius2));
    let world_radius = select(max(max_radius * 3.0 * splat_scale, 0.0001), 0.01, point_mode);

    let world_offset =
        uniforms.right.xyz * corner.x * world_radius +
        uniforms.up.xyz * corner.y * world_radius;

    var out: VertexOut;
    out.position = uniforms.view_proj * vec4<f32>(center + world_offset, 1.0);
    out.local = corner;
    out.color = vec4<f32>(input.color.rgb * opacity, opacity);
    return out;
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    let r2 = dot(input.local, input.local);
    if (r2 > 1.0) {
        discard;
    }
    let gaussian = exp(-2.0 * r2);
    return vec4<f32>(input.color.rgb * gaussian, input.color.a * gaussian);
}
