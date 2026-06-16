use std::mem::size_of;

use anyhow::Result;
use glam::{Mat4, Vec2, Vec3, Vec4};

use crate::{
    camera::Camera,
    cameras,
    cli::InspectArgs,
    loader,
    scene::{GaussianGpu, GaussianRaw},
};

const KERNEL_CUTOFF: f32 = 8.0;

pub fn run(args: InspectArgs) -> Result<()> {
    let scene = loader::load_scene(&args.model, args.max_splats)?;
    let f_rest_count = scene.raw.first().map(|raw| raw.f_rest.len()).unwrap_or(0);
    let gpu_bytes = scene.len() * size_of::<GaussianGpu>();
    let raw_bytes = scene.len() * estimated_raw_bytes_per_splat(f_rest_count);

    println!("model: {}", scene.source_label);
    println!("splats: {}", scene.len());
    println!("bounds min: {}", format_vec3(scene.bounds_min));
    println!("bounds max: {}", format_vec3(scene.bounds_max));
    println!("center: {}", format_vec3(scene.center));
    println!("radius: {:.6}", scene.radius);
    println!("view center: {}", format_vec3(scene.view_center));
    println!("view radius: {:.6}", scene.view_radius);
    println!("detected SH degree: {}", scene.detected_sh_degree());
    println!("f_rest coefficients per splat: {f_rest_count}");
    println!("estimated raw splat data: {}", format_bytes(raw_bytes));
    println!("packed GPU instance data: {}", format_bytes(gpu_bytes));

    if let Some(camera_index) = args.camera_index {
        let width = args.width.max(1);
        let height = args.height.max(1);
        let camera = make_camera(&args, &scene, camera_index, width, height);
        let stats = projected_radius_stats(
            &scene.gpu,
            &camera,
            width,
            height,
            args.splat_scale.clamp(0.05, 12.0),
            args.max_splat_radius.clamp(2.0, 1024.0),
        );
        print_projected_stats(camera_index, &stats);
    }

    Ok(())
}

fn make_camera(
    args: &InspectArgs,
    scene: &crate::scene::SplatScene,
    camera_index: usize,
    width: u32,
    height: u32,
) -> Camera {
    let aspect = width as f32 / height as f32;
    match cameras::load_preset_for_model(&args.model, scene, camera_index) {
        Ok(Some(preset)) => Camera::from_eye_target_up(
            preset.eye,
            preset.target,
            preset.up,
            scene.radius,
            aspect,
            preset.fovy_radians,
        ),
        Ok(None) | Err(_) => Camera::for_scene(scene.view_center, scene.view_radius, aspect),
    }
}

#[derive(Debug)]
struct ProjectedRadiusStats {
    visible_count: usize,
    clamped_count: usize,
    raw_radius_px: Vec<f32>,
    final_radius_px: Vec<f32>,
}

fn projected_radius_stats(
    splats: &[GaussianGpu],
    camera: &Camera,
    width: u32,
    height: u32,
    splat_scale: f32,
    max_splat_radius: f32,
) -> ProjectedRadiusStats {
    let view = camera.view();
    let view_proj = camera.view_projection();
    let tan_fovy = (camera.fovy_radians * 0.5).tan();
    let tan_fovx = tan_fovy * camera.aspect.max(0.001);
    let focal = Vec2::new(
        width as f32 / (2.0 * tan_fovx.max(0.001)),
        height as f32 / (2.0 * tan_fovy.max(0.001)),
    );
    let mut raw_radius_px = Vec::new();
    let mut final_radius_px = Vec::new();
    let mut clamped_count = 0;

    for splat in splats {
        let center = Vec3::new(
            splat.position_opacity[0],
            splat.position_opacity[1],
            splat.position_opacity[2],
        );
        let center_clip = view_proj * center.extend(1.0);
        if center_clip.w <= 0.001 {
            continue;
        }
        let center_view = view * center.extend(1.0);
        let depth = -center_view.z;
        let ndc_x = center_clip.x / center_clip.w;
        let ndc_y = center_clip.y / center_clip.w;
        if depth <= camera.z_near
            || depth >= camera.z_far
            || ndc_x.abs() > 1.35
            || ndc_y.abs() > 1.35
        {
            continue;
        }

        let axis0 = axis_from(splat.axis0_radius) * splat.axis0_radius[3] * splat_scale;
        let axis1 = axis_from(splat.axis1_radius) * splat.axis1_radius[3] * splat_scale;
        let axis2 = axis_from(splat.axis2_radius) * splat.axis2_radius[3] * splat_scale;
        let s0 = axis_screen_offset(view, center_view, focal, tan_fovx, tan_fovy, axis0);
        let s1 = axis_screen_offset(view, center_view, focal, tan_fovx, tan_fovy, axis1);
        let s2 = axis_screen_offset(view, center_view, focal, tan_fovx, tan_fovy, axis2);

        let mut cov_xx = Vec3::new(s0.x, s1.x, s2.x).length_squared() + 0.3;
        let mut cov_xy = Vec3::new(s0.x, s1.x, s2.x).dot(Vec3::new(s0.y, s1.y, s2.y));
        let mut cov_yy = Vec3::new(s0.y, s1.y, s2.y).length_squared() + 0.3;
        let raw_max_eigen = max_eigen(cov_xx, cov_xy, cov_yy);
        let raw_radius = (KERNEL_CUTOFF * raw_max_eigen).sqrt().max(2.0);
        let max_allowed_eigen = ((max_splat_radius.max(2.0) / 3.0).powi(2)).max(1.0);
        let covariance_scale = (max_allowed_eigen / raw_max_eigen.max(1.0)).min(1.0);
        if covariance_scale < 0.999 {
            clamped_count += 1;
        }
        cov_xx *= covariance_scale;
        cov_xy *= covariance_scale;
        cov_yy *= covariance_scale;
        let final_radius = (KERNEL_CUTOFF * max_eigen(cov_xx, cov_xy, cov_yy))
            .sqrt()
            .max(2.0)
            .min(max_splat_radius);
        raw_radius_px.push(raw_radius);
        final_radius_px.push(final_radius);
    }

    ProjectedRadiusStats {
        visible_count: raw_radius_px.len(),
        clamped_count,
        raw_radius_px,
        final_radius_px,
    }
}

fn axis_from(value: [f32; 4]) -> Vec3 {
    Vec3::new(value[0], value[1], value[2])
}

fn axis_screen_offset(
    view: Mat4,
    center_view: Vec4,
    focal: Vec2,
    tan_fovx: f32,
    tan_fovy: f32,
    axis_world: Vec3,
) -> Vec2 {
    let axis_view = view * axis_world.extend(0.0);
    let axis_cam = Vec3::new(axis_view.x, axis_view.y, -axis_view.z);
    let z = (-center_view.z).max(0.001);
    let lim_x = 1.3 * tan_fovx;
    let lim_y = 1.3 * tan_fovy;
    let x = (center_view.x / z).clamp(-lim_x, lim_x) * z;
    let y = (center_view.y / z).clamp(-lim_y, lim_y) * z;
    Vec2::new(
        focal.x / z * axis_cam.x - focal.x * x / (z * z) * axis_cam.z,
        focal.y / z * axis_cam.y - focal.y * y / (z * z) * axis_cam.z,
    )
}

fn max_eigen(cov_xx: f32, cov_xy: f32, cov_yy: f32) -> f32 {
    let trace = cov_xx + cov_yy;
    let diff = cov_xx - cov_yy;
    let disc = (diff * diff + 4.0 * cov_xy * cov_xy).max(0.0).sqrt();
    (0.5 * (trace + disc)).max(1.0)
}

fn print_projected_stats(camera_index: usize, stats: &ProjectedRadiusStats) {
    println!("projected camera index: {camera_index}");
    println!("projected visible splats: {}", stats.visible_count);
    println!(
        "projected radius-clamped splats: {} ({:.2}%)",
        stats.clamped_count,
        percentage(stats.clamped_count, stats.visible_count)
    );
    println!(
        "raw projected radius px q50/q90/q95/q99/q99.5/q99.9/max: {}",
        format_quantiles(&stats.raw_radius_px)
    );
    println!(
        "final projected radius px q50/q90/q95/q99/q99.5/q99.9/max: {}",
        format_quantiles(&stats.final_radius_px)
    );
}

fn estimated_raw_bytes_per_splat(f_rest_count: usize) -> usize {
    let fixed_floats = 3 + 3 + 1 + 3 + 4;
    let scalar_bytes = (fixed_floats + f_rest_count) * size_of::<f32>();
    scalar_bytes.max(size_of::<GaussianRaw>())
}

fn format_vec3(value: glam::Vec3) -> String {
    format!("({:.6}, {:.6}, {:.6})", value.x, value.y, value.z)
}

fn format_bytes(bytes: usize) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;

    let bytes = bytes as f64;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes / GIB)
    } else if bytes >= MIB {
        format!("{:.2} MiB", bytes / MIB)
    } else if bytes >= KIB {
        format!("{:.2} KiB", bytes / KIB)
    } else {
        format!("{bytes:.0} B")
    }
}

fn percentage(part: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        part as f32 / total as f32 * 100.0
    }
}

fn format_quantiles(values: &[f32]) -> String {
    if values.is_empty() {
        return "n/a".into();
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f32::total_cmp);
    [0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0]
        .into_iter()
        .map(|q| format!("{:.3}", quantile_sorted(&sorted, q)))
        .collect::<Vec<_>>()
        .join(" / ")
}

fn quantile_sorted(sorted: &[f32], q: f32) -> f32 {
    let last = sorted.len() - 1;
    let index = (q.clamp(0.0, 1.0) * last as f32).round() as usize;
    sorted[index.min(last)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_byte_counts() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KiB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MiB");
    }

    #[test]
    fn formats_quantiles() {
        assert_eq!(format_quantiles(&[]), "n/a");
        assert_eq!(
            format_quantiles(&[1.0, 10.0, 5.0]),
            "5.000 / 10.000 / 10.000 / 10.000 / 10.000 / 10.000 / 10.000"
        );
    }
}
