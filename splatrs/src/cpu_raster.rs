use std::thread;

use anyhow::Result;
use glam::{Mat4, Vec2, Vec3, Vec4};

use crate::{
    camera::Camera,
    renderer::RenderOptions,
    scene::{DepthSort, GaussianGpu, SplatScene},
};

const KERNEL_CUTOFF: f32 = 8.0;
const TILE_SIZE: usize = 16;
const MIN_ALPHA: f32 = 1.0 / 255.0;
const TRANSMITTANCE_EPSILON: f32 = 0.0001;

#[derive(Clone, Copy, Debug)]
struct CpuSplat {
    center_px: Vec2,
    conic: [f32; 3],
    color: [f32; 3],
    opacity: f32,
    bbox_min: [usize; 2],
    bbox_max: [usize; 2],
}

pub fn render_tile_cpu(
    scene: &SplatScene,
    camera: &Camera,
    options: RenderOptions,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    let width = width.max(1);
    let height = height.max(1);
    let sorted = scene.sorted_gpu_for_camera(
        camera.view(),
        camera.view_projection(),
        camera.eye(),
        options.sh_degree,
        camera.z_near,
        camera.z_far,
        DepthSort::FrontToBack,
    );

    let projected = project_splats(&sorted, camera, options, width, height);
    let tile_columns = (width as usize).div_ceil(TILE_SIZE);
    let tile_rows = (height as usize).div_ceil(TILE_SIZE);
    let tiles = bin_splats(&projected, tile_columns, tile_rows);

    let mut pixels = vec![0; width as usize * height as usize * 4];
    render_pixels(
        &projected,
        &tiles,
        tile_columns,
        width as usize,
        options.background,
        &mut pixels,
    )?;
    Ok(pixels)
}

fn project_splats(
    splats: &[GaussianGpu],
    camera: &Camera,
    options: RenderOptions,
    width: u32,
    height: u32,
) -> Vec<CpuSplat> {
    let view = camera.view();
    let view_proj = camera.view_projection();
    let tan_fovy = (camera.fovy_radians * 0.5).tan();
    let tan_fovx = tan_fovy * camera.aspect.max(0.001);
    let focal_x = width as f32 / (2.0 * tan_fovx.max(0.001));
    let focal_y = height as f32 / (2.0 * tan_fovy.max(0.001));
    let mut projected = Vec::with_capacity(splats.len());

    for splat in splats {
        if let Some(cpu_splat) = project_splat(
            *splat, view, view_proj, focal_x, focal_y, tan_fovx, tan_fovy, options, width, height,
        ) {
            projected.push(cpu_splat);
        }
    }

    projected
}

#[allow(clippy::too_many_arguments)]
fn project_splat(
    splat: GaussianGpu,
    view: Mat4,
    view_proj: Mat4,
    focal_x: f32,
    focal_y: f32,
    tan_fovx: f32,
    tan_fovy: f32,
    options: RenderOptions,
    width: u32,
    height: u32,
) -> Option<CpuSplat> {
    let center = splat.position();
    let center_clip = view_proj * center.extend(1.0);
    if center_clip.w <= 0.001 {
        return None;
    }

    let mut opacity = (splat.position_opacity[3] * options.opacity_scale).clamp(0.0, 1.0);
    if opacity <= 0.0 {
        return None;
    }

    let point_mode = options.point_mode;
    let splat_scale = options.splat_scale;
    let center_view = view * center.extend(1.0);
    let (mut cov_xx, mut cov_xy, mut cov_yy) = if point_mode {
        (4.0, 0.0, 4.0)
    } else {
        let axis0 = axis_from(splat.axis0_radius) * splat.axis0_radius[3] * splat_scale;
        let axis1 = axis_from(splat.axis1_radius) * splat.axis1_radius[3] * splat_scale;
        let axis2 = axis_from(splat.axis2_radius) * splat.axis2_radius[3] * splat_scale;
        let s0 = axis_screen_offset(
            center_view,
            axis0,
            view,
            focal_x,
            focal_y,
            tan_fovx,
            tan_fovy,
        );
        let s1 = axis_screen_offset(
            center_view,
            axis1,
            view,
            focal_x,
            focal_y,
            tan_fovx,
            tan_fovy,
        );
        let s2 = axis_screen_offset(
            center_view,
            axis2,
            view,
            focal_x,
            focal_y,
            tan_fovx,
            tan_fovy,
        );

        (
            Vec3::new(s0.x, s1.x, s2.x).length_squared() + 0.3,
            Vec3::new(s0.x, s1.x, s2.x).dot(Vec3::new(s0.y, s1.y, s2.y)),
            Vec3::new(s0.y, s1.y, s2.y).length_squared() + 0.3,
        )
    };

    let max_quad_radius = if point_mode {
        8.0
    } else {
        options.max_splat_radius.max(2.0)
    };
    let raw_max_eigen = max_eigen(cov_xx, cov_xy, cov_yy).max(1.0);
    let max_allowed_eigen = ((max_quad_radius / 3.0) * (max_quad_radius / 3.0)).max(1.0);
    let covariance_scale = (max_allowed_eigen / raw_max_eigen).min(1.0);
    cov_xx *= covariance_scale;
    cov_xy *= covariance_scale;
    cov_yy *= covariance_scale;
    opacity *= covariance_scale * covariance_scale;
    if opacity < MIN_ALPHA {
        return None;
    }

    let max_eigen = max_eigen(cov_xx, cov_xy, cov_yy).max(1.0);
    let quad_radius = (KERNEL_CUTOFF * max_eigen)
        .sqrt()
        .clamp(2.0, max_quad_radius);
    let det = (cov_xx * cov_yy - cov_xy * cov_xy).max(0.0001);
    let ndc_x = center_clip.x / center_clip.w;
    let ndc_y = center_clip.y / center_clip.w;
    let center_px = Vec2::new(
        (ndc_x * 0.5 + 0.5) * width as f32,
        (0.5 - ndc_y * 0.5) * height as f32,
    );

    let min_x = (center_px.x - quad_radius).floor().max(0.0) as usize;
    let min_y = (center_px.y - quad_radius).floor().max(0.0) as usize;
    let max_x = (center_px.x + quad_radius).ceil().min(width as f32) as usize;
    let max_y = (center_px.y + quad_radius).ceil().min(height as f32) as usize;
    if min_x >= max_x || min_y >= max_y {
        return None;
    }

    Some(CpuSplat {
        center_px,
        conic: [cov_yy / det, -cov_xy / det, cov_xx / det],
        color: [splat.color[0], splat.color[1], splat.color[2]],
        opacity,
        bbox_min: [min_x, min_y],
        bbox_max: [max_x, max_y],
    })
}

fn axis_screen_offset(
    center_view: Vec4,
    axis_world: Vec3,
    view: Mat4,
    focal_x: f32,
    focal_y: f32,
    tan_fovx: f32,
    tan_fovy: f32,
) -> Vec2 {
    let axis_view = view * axis_world.extend(0.0);
    let axis_cam = Vec3::new(axis_view.x, axis_view.y, -axis_view.z);
    let z = (-center_view.z).max(0.001);
    let x = (center_view.x / z).clamp(-1.3 * tan_fovx, 1.3 * tan_fovx) * z;
    let y = (center_view.y / z).clamp(-1.3 * tan_fovy, 1.3 * tan_fovy) * z;

    Vec2::new(
        focal_x / z * axis_cam.x - focal_x * x / (z * z) * axis_cam.z,
        focal_y / z * axis_cam.y - focal_y * y / (z * z) * axis_cam.z,
    )
}

fn bin_splats(splats: &[CpuSplat], tile_columns: usize, tile_rows: usize) -> Vec<Vec<usize>> {
    let mut tiles = vec![Vec::new(); tile_columns * tile_rows];
    for (index, splat) in splats.iter().enumerate() {
        let min_tile_x = splat.bbox_min[0] / TILE_SIZE;
        let min_tile_y = splat.bbox_min[1] / TILE_SIZE;
        let max_tile_x = (splat.bbox_max[0] - 1) / TILE_SIZE;
        let max_tile_y = (splat.bbox_max[1] - 1) / TILE_SIZE;
        for tile_y in min_tile_y..=max_tile_y.min(tile_rows - 1) {
            for tile_x in min_tile_x..=max_tile_x.min(tile_columns - 1) {
                tiles[tile_y * tile_columns + tile_x].push(index);
            }
        }
    }
    tiles
}

fn render_pixels(
    splats: &[CpuSplat],
    tiles: &[Vec<usize>],
    tile_columns: usize,
    width: usize,
    background: [f32; 3],
    pixels: &mut [u8],
) -> Result<()> {
    let row_bytes = width * 4;
    let height = pixels.len() / row_bytes;
    let workers = thread::available_parallelism()
        .map_or(1, usize::from)
        .min(height.max(1));
    let rows_per_worker = height.div_ceil(workers);

    thread::scope(|scope| {
        for (worker_index, chunk) in pixels.chunks_mut(rows_per_worker * row_bytes).enumerate() {
            let start_y = worker_index * rows_per_worker;
            scope.spawn(move || {
                render_row_chunk(
                    splats,
                    tiles,
                    tile_columns,
                    width,
                    start_y,
                    background,
                    chunk,
                );
            });
        }
    });

    Ok(())
}

fn render_row_chunk(
    splats: &[CpuSplat],
    tiles: &[Vec<usize>],
    tile_columns: usize,
    width: usize,
    start_y: usize,
    background: [f32; 3],
    pixels: &mut [u8],
) {
    let row_bytes = width * 4;
    let local_height = pixels.len() / row_bytes;
    for local_y in 0..local_height {
        let y = start_y + local_y;
        let tile_y = y / TILE_SIZE;
        for x in 0..width {
            let tile_x = x / TILE_SIZE;
            let tile = &tiles[tile_y * tile_columns + tile_x];
            let color = render_pixel(splats, tile, x as f32 + 0.5, y as f32 + 0.5, background);
            let offset = local_y * row_bytes + x * 4;
            pixels[offset] = (linear_to_srgb(color[0]) * 255.0).round() as u8;
            pixels[offset + 1] = (linear_to_srgb(color[1]) * 255.0).round() as u8;
            pixels[offset + 2] = (linear_to_srgb(color[2]) * 255.0).round() as u8;
            pixels[offset + 3] = 255;
        }
    }
}

fn render_pixel(
    splats: &[CpuSplat],
    tile: &[usize],
    pixel_x: f32,
    pixel_y: f32,
    background: [f32; 3],
) -> [f32; 3] {
    let mut transmittance = 1.0;
    let mut color = [0.0; 3];

    for &index in tile {
        let splat = splats[index];
        if pixel_x < splat.bbox_min[0] as f32
            || pixel_x >= splat.bbox_max[0] as f32
            || pixel_y < splat.bbox_min[1] as f32
            || pixel_y >= splat.bbox_max[1] as f32
        {
            continue;
        }

        let dx = pixel_x - splat.center_px.x;
        let dy = splat.center_px.y - pixel_y;
        let q =
            splat.conic[0] * dx * dx + 2.0 * splat.conic[1] * dx * dy + splat.conic[2] * dy * dy;
        if q > KERNEL_CUTOFF {
            continue;
        }

        let alpha = (splat.opacity * (-0.5 * q).exp()).min(0.99);
        if alpha < MIN_ALPHA {
            continue;
        }

        let next_transmittance = transmittance * (1.0 - alpha);
        if next_transmittance < TRANSMITTANCE_EPSILON {
            break;
        }

        for channel in 0..3 {
            color[channel] += splat.color[channel] * alpha * transmittance;
        }
        transmittance = next_transmittance;
    }

    [
        (color[0] + transmittance * background[0]).clamp(0.0, 1.0),
        (color[1] + transmittance * background[1]).clamp(0.0, 1.0),
        (color[2] + transmittance * background[2]).clamp(0.0, 1.0),
    ]
}

fn axis_from(axis_radius: [f32; 4]) -> Vec3 {
    Vec3::new(axis_radius[0], axis_radius[1], axis_radius[2])
}

fn max_eigen(cov_xx: f32, cov_xy: f32, cov_yy: f32) -> f32 {
    let trace = cov_xx + cov_yy;
    let diff = cov_xx - cov_yy;
    let disc = (diff * diff + 4.0 * cov_xy * cov_xy).max(0.0).sqrt();
    0.5 * (trace + disc)
}

fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{GaussianRaw, SplatScene};

    #[test]
    fn cpu_tile_renderer_produces_nonblank_frame() {
        let raw = GaussianRaw {
            position: Vec3::new(0.0, 0.0, -3.0),
            f_dc: [2.0, 0.0, 0.0],
            f_rest: Vec::new(),
            opacity_logit: 4.0,
            log_scale: Vec3::splat(-1.2),
            rotation_raw: Vec4::new(1.0, 0.0, 0.0, 0.0),
        };
        let scene = SplatScene::from_raw(vec![raw], "test".into());
        let camera = Camera::from_eye_target_up(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::Y,
            1.0,
            1.0,
            60.0_f32.to_radians(),
        );

        let pixels = render_tile_cpu(&scene, &camera, RenderOptions::default(), 64, 64).unwrap();

        assert!(pixels.chunks_exact(4).any(|pixel| pixel[0] > 80));
    }

    #[test]
    fn srgb_transfer_keeps_bounds() {
        assert_eq!(linear_to_srgb(0.0), 0.0);
        assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-6);
        assert!(linear_to_srgb(0.5) > 0.5);
    }
}
