struct Uniforms {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    viewport: vec4<f32>,
    focal: vec4<f32>,
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

fn axis_screen_offset(center_view: vec4<f32>, axis_world: vec3<f32>) -> vec2<f32> {
    let axis_view = uniforms.view * vec4<f32>(axis_world, 0.0);
    let axis_cam = vec3<f32>(axis_view.x, axis_view.y, -axis_view.z);
    let z = max(-center_view.z, 0.001);

    let lim_x = 1.3 * uniforms.focal.z;
    let lim_y = 1.3 * uniforms.focal.w;
    let x = clamp(center_view.x / z, -lim_x, lim_x) * z;
    let y = clamp(center_view.y / z, -lim_y, lim_y) * z;

    return vec2<f32>(
        uniforms.focal.x / z * axis_cam.x - uniforms.focal.x * x / (z * z) * axis_cam.z,
        uniforms.focal.y / z * axis_cam.y - uniforms.focal.y * y / (z * z) * axis_cam.z,
    );
}

@vertex
fn vs_main(input: VertexIn) -> VertexOut {
    let corner = quad_corner(input.vertex_index);
    let center = input.position_opacity.xyz;
    var opacity = clamp(input.position_opacity.w * uniforms.options.x, 0.0, 1.0);
    let point_mode = uniforms.options.y > 0.5;
    let splat_scale = uniforms.options.z;
    let max_splat_radius_option = uniforms.options.w;

    let center_clip = uniforms.view_proj * vec4<f32>(center, 1.0);
    if (center_clip.w <= 0.001) {
        var out: VertexOut;
        out.position = vec4<f32>(2.0, 2.0, 1.0, 1.0);
        out.delta_px = vec2<f32>(0.0, 0.0);
        out.color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.conic = vec4<f32>(1.0, 0.0, 1.0, 0.0);
        return out;
    }
    let center_view = uniforms.view * vec4<f32>(center, 1.0);

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
        let s0 = axis_screen_offset(center_view, axis0);
        let s1 = axis_screen_offset(center_view, axis1);
        let s2 = axis_screen_offset(center_view, axis2);

        cov_xx = dot(vec3<f32>(s0.x, s1.x, s2.x), vec3<f32>(s0.x, s1.x, s2.x)) + 0.3;
        cov_xy = dot(vec3<f32>(s0.x, s1.x, s2.x), vec3<f32>(s0.y, s1.y, s2.y));
        cov_yy = dot(vec3<f32>(s0.y, s1.y, s2.y), vec3<f32>(s0.y, s1.y, s2.y)) + 0.3;
    }

    let max_quad_radius = select(max(max_splat_radius_option, 2.0), 8.0, point_mode);
    let raw_trace = cov_xx + cov_yy;
    let raw_diff = cov_xx - cov_yy;
    let raw_eigen_disc = sqrt(max(raw_diff * raw_diff + 4.0 * cov_xy * cov_xy, 0.0));
    let raw_max_eigen = max(0.5 * (raw_trace + raw_eigen_disc), 1.0);
    let max_allowed_eigen = max((max_quad_radius / 3.0) * (max_quad_radius / 3.0), 1.0);
    let covariance_scale = min(1.0, max_allowed_eigen / raw_max_eigen);
    cov_xx = cov_xx * covariance_scale;
    cov_xy = cov_xy * covariance_scale;
    cov_yy = cov_yy * covariance_scale;
    // Keep oversized splats from retaining full alpha after their footprint is clamped.
    opacity = opacity * covariance_scale;

    let trace = cov_xx + cov_yy;
    let diff = cov_xx - cov_yy;
    let eigen_disc = sqrt(max(diff * diff + 4.0 * cov_xy * cov_xy, 0.0));
    let max_eigen = max(0.5 * (trace + eigen_disc), 1.0);
    let kernel_cutoff = 8.0;
    let quad_radius = min(max(sqrt(kernel_cutoff * max_eigen), 2.0), max_quad_radius);
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
    if (q > 8.0) {
        discard;
    }
    let gaussian = exp(-0.5 * q);
    let alpha = min(input.color.a * gaussian, 0.99);
    if (alpha < 0.0039215689) {
        discard;
    }
    let base_color = input.color.rgb / max(input.color.a, 0.000001);
    return vec4<f32>(base_color * alpha, alpha);
}
