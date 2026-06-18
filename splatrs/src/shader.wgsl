struct Uniforms {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    viewport: vec4<f32>,
    focal: vec4<f32>,
    options: vec4<f32>,
    post: vec4<f32>,
    quality: vec4<f32>,
    alpha: vec4<f32>,
    color_options: vec4<f32>,
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

fn axis_camera(axis_world: vec3<f32>) -> vec3<f32> {
    let axis_view = uniforms.view * vec4<f32>(axis_world, 0.0);
    return vec3<f32>(axis_view.x, axis_view.y, -axis_view.z);
}

fn axis_screen_offset_from_camera(center_view: vec4<f32>, axis_cam: vec3<f32>) -> vec2<f32> {
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

fn axis_screen_offset(center_view: vec4<f32>, axis_world: vec3<f32>) -> vec2<f32> {
    return axis_screen_offset_from_camera(center_view, axis_camera(axis_world));
}

fn covariance_mul(
    cov0: vec3<f32>,
    cov1: vec3<f32>,
    cov2: vec3<f32>,
    value: vec3<f32>,
) -> vec3<f32> {
    return vec3<f32>(dot(cov0, value), dot(cov1, value), dot(cov2, value));
}

fn covariance_screen_footprint(
    center_view: vec4<f32>,
    axis0_world: vec3<f32>,
    axis1_world: vec3<f32>,
    axis2_world: vec3<f32>,
) -> vec3<f32> {
    let axis0 = axis_camera(axis0_world);
    let axis1 = axis_camera(axis1_world);
    let axis2 = axis_camera(axis2_world);
    let cov00 = axis0.x * axis0.x + axis1.x * axis1.x + axis2.x * axis2.x;
    let cov01 = axis0.x * axis0.y + axis1.x * axis1.y + axis2.x * axis2.y;
    let cov02 = axis0.x * axis0.z + axis1.x * axis1.z + axis2.x * axis2.z;
    let cov11 = axis0.y * axis0.y + axis1.y * axis1.y + axis2.y * axis2.y;
    let cov12 = axis0.y * axis0.z + axis1.y * axis1.z + axis2.y * axis2.z;
    let cov22 = axis0.z * axis0.z + axis1.z * axis1.z + axis2.z * axis2.z;
    let cov0 = vec3<f32>(cov00, cov01, cov02);
    let cov1 = vec3<f32>(cov01, cov11, cov12);
    let cov2 = vec3<f32>(cov02, cov12, cov22);

    let z = max(-center_view.z, 0.001);
    let lim_x = 1.3 * uniforms.focal.z;
    let lim_y = 1.3 * uniforms.focal.w;
    let x = clamp(center_view.x / z, -lim_x, lim_x) * z;
    let y = clamp(center_view.y / z, -lim_y, lim_y) * z;
    let inv_z = 1.0 / z;
    let inv_z2 = inv_z * inv_z;
    let jac_x = vec3<f32>(uniforms.focal.x * inv_z, 0.0, -uniforms.focal.x * x * inv_z2);
    let jac_y = vec3<f32>(0.0, uniforms.focal.y * inv_z, -uniforms.focal.y * y * inv_z2);
    let cov_xx = dot(jac_x, covariance_mul(cov0, cov1, cov2, jac_x));
    let cov_xy = dot(jac_x, covariance_mul(cov0, cov1, cov2, jac_y));
    let cov_yy = dot(jac_y, covariance_mul(cov0, cov1, cov2, jac_y));

    return vec3<f32>(cov_xx, cov_xy, cov_yy);
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
        var base_cov: vec3<f32>;
        if (uniforms.quality.x > 0.5) {
            base_cov = covariance_screen_footprint(center_view, axis0, axis1, axis2);
        } else {
            let s0 = axis_screen_offset(center_view, axis0);
            let s1 = axis_screen_offset(center_view, axis1);
            let s2 = axis_screen_offset(center_view, axis2);
            base_cov = vec3<f32>(
                dot(vec3<f32>(s0.x, s1.x, s2.x), vec3<f32>(s0.x, s1.x, s2.x)),
                dot(vec3<f32>(s0.x, s1.x, s2.x), vec3<f32>(s0.y, s1.y, s2.y)),
                dot(vec3<f32>(s0.y, s1.y, s2.y), vec3<f32>(s0.y, s1.y, s2.y)),
            );
        }

        let base_cov_xx = base_cov.x;
        let base_cov_xy = base_cov.y;
        let base_cov_yy = base_cov.z;
        let det_before_lowpass = max(base_cov_xx * base_cov_yy - base_cov_xy * base_cov_xy, 0.0001);
        let lowpass = max(uniforms.quality.y, 0.0);

        cov_xx = base_cov_xx + lowpass;
        cov_xy = base_cov_xy;
        cov_yy = base_cov_yy + lowpass;

        if (uniforms.post.z > 0.5) {
            let det_after_lowpass = max(cov_xx * cov_yy - cov_xy * cov_xy, 0.0001);
            let lowpass_alpha_scale = sqrt(max(0.000025, det_before_lowpass / det_after_lowpass));
            opacity = opacity * lowpass_alpha_scale;
        }
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
    let radius_alpha = uniforms.quality.w;
    if (radius_alpha < 0.5) {
        opacity = opacity * covariance_scale * covariance_scale;
    } else if (radius_alpha < 1.5) {
        opacity = opacity * covariance_scale;
    }

    let trace = cov_xx + cov_yy;
    let diff = cov_xx - cov_yy;
    let eigen_disc = sqrt(max(diff * diff + 4.0 * cov_xy * cov_xy, 0.0));
    let max_eigen = max(0.5 * (trace + eigen_disc), 1.0);
    let kernel_cutoff = max(uniforms.quality.z, 0.5);
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

fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + vec3<f32>(b))) / (color * (c * color + vec3<f32>(d)) + vec3<f32>(e)), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn display_color(color: vec3<f32>) -> vec3<f32> {
    let color_max = max(uniforms.color_options.x, 0.001);
    let saturation = clamp(uniforms.color_options.y, 0.0, 2.0);
    let clamped = clamp(color, vec3<f32>(0.0), vec3<f32>(color_max));
    let luma = dot(clamped, vec3<f32>(0.2126, 0.7152, 0.0722));
    let shaped = vec3<f32>(luma) + (clamped - vec3<f32>(luma)) * saturation;
    let exposed = max(shaped * uniforms.post.x, vec3<f32>(0.0));
    if (uniforms.post.y > 1.5) {
        return aces_tonemap(exposed);
    }
    if (uniforms.post.y > 0.5) {
        return exposed / (vec3<f32>(1.0) + exposed);
    }
    return exposed;
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    let q =
        input.conic.x * input.delta_px.x * input.delta_px.x +
        2.0 * input.conic.y * input.delta_px.x * input.delta_px.y +
        input.conic.z * input.delta_px.y * input.delta_px.y;
    if (q > max(uniforms.quality.z, 0.5)) {
        discard;
    }
    let gaussian = exp(-0.5 * q);
    let alpha = min(input.color.a * gaussian, clamp(uniforms.alpha.y, 0.0, 1.0));
    if (alpha < clamp(uniforms.alpha.x, 0.0, 1.0)) {
        discard;
    }
    let base_color = display_color(input.color.rgb / max(input.color.a, 0.000001));
    return vec4<f32>(base_color * alpha, alpha);
}
