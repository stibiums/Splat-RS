struct Uniforms {
    view_proj: mat4x4<f32>,
    viewport: vec4<f32>,
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
    @location(0) delta_px: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) conic: vec4<f32>,
};

fn quad_corner(index: u32) -> vec2<f32> {
    switch index {
        case 0u: { return vec2<f32>(-1.0, -1.0); }
        case 1u: { return vec2<f32>( 1.0, -1.0); }
        case 2u: { return vec2<f32>(-1.0,  1.0); }
        default: { return vec2<f32>( 1.0,  1.0); }
    }
}

fn axis_screen_offset(center: vec3<f32>, axis: vec3<f32>, center_ndc: vec2<f32>) -> vec2<f32> {
    let axis_clip = uniforms.view_proj * vec4<f32>(center + axis, 1.0);
    if (axis_clip.w <= 0.001) {
        return vec2<f32>(0.0, 0.0);
    }
    return (axis_clip.xy / axis_clip.w - center_ndc) * uniforms.viewport.xy * 0.5;
}

@vertex
fn vs_main(input: VertexIn) -> VertexOut {
    let corner = quad_corner(input.vertex_index);
    let center = input.position_opacity.xyz;
    let opacity = clamp(input.position_opacity.w * uniforms.options.x, 0.0, 1.0);
    let point_mode = uniforms.options.y > 0.5;
    let splat_scale = uniforms.options.z;

    let center_clip = uniforms.view_proj * vec4<f32>(center, 1.0);
    if (center_clip.w <= 0.001) {
        var out: VertexOut;
        out.position = vec4<f32>(2.0, 2.0, 1.0, 1.0);
        out.delta_px = vec2<f32>(0.0, 0.0);
        out.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.conic = vec4<f32>(1.0, 0.0, 1.0, 0.0);
        return out;
    }
    let center_ndc = center_clip.xy / center_clip.w;

    var cov_xx: f32;
    var cov_xy: f32;
    var cov_yy: f32;

    if (point_mode) {
        cov_xx = 4.0;
        cov_xy = 0.0;
        cov_yy = 4.0;
    } else {
        let axis0 = input.axis0_radius.xyz * input.axis0_radius.w * splat_scale;
        let axis1 = input.axis1_radius.xyz * input.axis1_radius.w * splat_scale;
        let axis2 = input.axis2_radius.xyz * input.axis2_radius.w * splat_scale;
        let s0 = axis_screen_offset(center, axis0, center_ndc);
        let s1 = axis_screen_offset(center, axis1, center_ndc);
        let s2 = axis_screen_offset(center, axis2, center_ndc);

        cov_xx = dot(vec3<f32>(s0.x, s1.x, s2.x), vec3<f32>(s0.x, s1.x, s2.x)) + 0.35;
        cov_xy = dot(vec3<f32>(s0.x, s1.x, s2.x), vec3<f32>(s0.y, s1.y, s2.y));
        cov_yy = dot(vec3<f32>(s0.y, s1.y, s2.y), vec3<f32>(s0.y, s1.y, s2.y)) + 0.35;
    }

    let trace = cov_xx + cov_yy;
    let diff = cov_xx - cov_yy;
    let eigen_disc = sqrt(max(diff * diff + 4.0 * cov_xy * cov_xy, 0.0));
    let max_eigen = max(0.5 * (trace + eigen_disc), 1.0);
    let max_quad_radius = select(96.0, 8.0, point_mode);
    let quad_radius = min(max(3.0 * sqrt(max_eigen), 2.0), max_quad_radius);
    let delta_px = corner * quad_radius;
    let det = max(cov_xx * cov_yy - cov_xy * cov_xy, 0.0001);
    let clip_xy = center_clip.xy + delta_px / uniforms.viewport.xy * 2.0 * center_clip.w;

    var out: VertexOut;
    out.position = vec4<f32>(clip_xy, center_clip.z, center_clip.w);
    out.delta_px = delta_px;
    out.color = vec4<f32>(input.color.rgb * opacity, opacity);
    out.conic = vec4<f32>(cov_yy / det, -cov_xy / det, cov_xx / det, 0.0);
    return out;
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    let q =
        input.conic.x * input.delta_px.x * input.delta_px.x +
        2.0 * input.conic.y * input.delta_px.x * input.delta_px.y +
        input.conic.z * input.delta_px.y * input.delta_px.y;
    if (q > 18.0) {
        discard;
    }
    let gaussian = exp(-0.5 * q);
    return vec4<f32>(input.color.rgb * gaussian, input.color.a * gaussian);
}
