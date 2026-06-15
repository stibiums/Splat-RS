use glam::{Mat4, Quat, Vec3, Vec4};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DepthSort {
    BackToFront,
    FrontToBack,
}

#[derive(Clone, Debug)]
pub struct GaussianRaw {
    pub position: Vec3,
    pub f_dc: [f32; 3],
    pub f_rest: Vec<f32>,
    pub opacity_logit: f32,
    pub log_scale: Vec3,
    pub rotation_raw: Vec4,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GaussianGpu {
    pub position_opacity: [f32; 4],
    pub color: [f32; 4],
    pub axis0_radius: [f32; 4],
    pub axis1_radius: [f32; 4],
    pub axis2_radius: [f32; 4],
}

impl GaussianGpu {
    pub fn from_raw(raw: &GaussianRaw) -> Self {
        let opacity = sigmoid(raw.opacity_logit);
        let scale = raw.log_scale.exp();
        let rotation = normalize_graphdeco_quat(raw.rotation_raw);
        let color = sh_dc_to_rgb(raw.f_dc);

        let axis0 = rotation * Vec3::X;
        let axis1 = rotation * Vec3::Y;
        let axis2 = rotation * Vec3::Z;

        Self {
            position_opacity: [raw.position.x, raw.position.y, raw.position.z, opacity],
            color: [color.x, color.y, color.z, 1.0],
            axis0_radius: [axis0.x, axis0.y, axis0.z, scale.x],
            axis1_radius: [axis1.x, axis1.y, axis1.z, scale.y],
            axis2_radius: [axis2.x, axis2.y, axis2.z, scale.z],
        }
    }

    pub fn with_color(mut self, color: Vec3) -> Self {
        self.color = [color.x, color.y, color.z, 1.0];
        self
    }

    pub fn position(&self) -> Vec3 {
        Vec3::new(
            self.position_opacity[0],
            self.position_opacity[1],
            self.position_opacity[2],
        )
    }
}

#[derive(Clone, Debug)]
pub struct SplatScene {
    pub raw: Vec<GaussianRaw>,
    pub gpu: Vec<GaussianGpu>,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub center: Vec3,
    pub radius: f32,
    pub view_center: Vec3,
    pub view_radius: f32,
    pub source_label: String,
}

impl SplatScene {
    pub fn from_raw(raw: Vec<GaussianRaw>, source_label: String) -> Self {
        let gpu: Vec<_> = raw.iter().map(GaussianGpu::from_raw).collect();
        let (bounds_min, bounds_max) = bounds_for(&raw);
        let center = (bounds_min + bounds_max) * 0.5;
        let radius = raw
            .iter()
            .map(|s| s.position.distance(center))
            .fold(0.0, f32::max)
            .max(0.001);
        let (view_min, view_max) = robust_bounds_for(&raw);
        let view_center = (view_min + view_max) * 0.5;
        let view_radius = (view_max.distance(view_min) * 0.5).max(0.001);

        Self {
            raw,
            gpu,
            bounds_min,
            bounds_max,
            center,
            radius,
            view_center,
            view_radius,
            source_label,
        }
    }

    pub fn len(&self) -> usize {
        self.gpu.len()
    }

    pub fn detected_sh_degree(&self) -> u32 {
        let Some(first) = self.raw.first() else {
            return 0;
        };
        available_sh_degree(first.f_rest.len())
    }

    pub fn sorted_gpu_from_eye(&self, eye: Vec3) -> Vec<GaussianGpu> {
        self.sorted_gpu_from_eye_with_sh(eye, 0)
    }

    pub fn sorted_gpu_from_eye_with_sh(&self, eye: Vec3, sh_degree: u32) -> Vec<GaussianGpu> {
        let mut order: Vec<usize> = (0..self.gpu.len()).collect();
        order.sort_unstable_by(|&a, &b| {
            let da = self.gpu[a].position().distance_squared(eye);
            let db = self.gpu[b].position().distance_squared(eye);
            db.total_cmp(&da)
        });
        order
            .into_iter()
            .map(|index| {
                let raw = &self.raw[index];
                let view_dir = (self.gpu[index].position() - eye).normalize_or_zero();
                let color = sh_to_rgb(raw.f_dc, &raw.f_rest, view_dir, sh_degree);
                self.gpu[index].with_color(color)
            })
            .collect()
    }

    pub fn sorted_gpu_for_camera(
        &self,
        view: Mat4,
        view_proj: Mat4,
        eye: Vec3,
        sh_degree: u32,
        near: f32,
        far: f32,
        depth_sort: DepthSort,
    ) -> Vec<GaussianGpu> {
        let mut visible = self
            .gpu
            .iter()
            .enumerate()
            .filter_map(|(index, splat)| {
                let position = splat.position();
                let view_pos = view.transform_point3(position);
                let depth = -view_pos.z;
                let clip = view_proj * position.extend(1.0);
                if clip.w <= 0.001 {
                    return None;
                }
                let ndc_x = clip.x / clip.w;
                let ndc_y = clip.y / clip.w;
                let inside_margin = ndc_x.abs() <= 1.35 && ndc_y.abs() <= 1.35;
                (depth > near && depth < far && inside_margin).then_some((index, depth))
            })
            .collect::<Vec<_>>();

        match depth_sort {
            DepthSort::BackToFront => {
                visible.sort_unstable_by(|(_, depth_a), (_, depth_b)| depth_b.total_cmp(depth_a));
            }
            DepthSort::FrontToBack => {
                visible.sort_unstable_by(|(_, depth_a), (_, depth_b)| depth_a.total_cmp(depth_b));
            }
        }
        visible
            .into_iter()
            .map(|(index, _)| {
                let raw = &self.raw[index];
                let view_dir = (self.gpu[index].position() - eye).normalize_or_zero();
                let color = sh_to_rgb(raw.f_dc, &raw.f_rest, view_dir, sh_degree);
                self.gpu[index].with_color(color)
            })
            .collect()
    }
}

fn bounds_for(raw: &[GaussianRaw]) -> (Vec3, Vec3) {
    raw.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(min, max), splat| (min.min(splat.position), max.max(splat.position)),
    )
}

fn robust_bounds_for(raw: &[GaussianRaw]) -> (Vec3, Vec3) {
    if raw.len() < 64 {
        return bounds_for(raw);
    }

    let mut xs = Vec::with_capacity(raw.len());
    let mut ys = Vec::with_capacity(raw.len());
    let mut zs = Vec::with_capacity(raw.len());
    for splat in raw {
        xs.push(splat.position.x);
        ys.push(splat.position.y);
        zs.push(splat.position.z);
    }

    xs.sort_unstable_by(f32::total_cmp);
    ys.sort_unstable_by(f32::total_cmp);
    zs.sort_unstable_by(f32::total_cmp);

    let trim = (raw.len() / 100).max(1);
    let lo = trim.min(raw.len() - 1);
    let hi = raw.len().saturating_sub(trim + 1).max(lo);

    (
        Vec3::new(xs[lo], ys[lo], zs[lo]),
        Vec3::new(xs[hi], ys[hi], zs[hi]),
    )
}

pub fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        let z = (-value).exp();
        1.0 / (1.0 + z)
    } else {
        let z = value.exp();
        z / (1.0 + z)
    }
}

pub fn normalize_graphdeco_quat(raw: Vec4) -> Quat {
    let len = raw.length();
    let q = if len > f32::EPSILON {
        raw / len
    } else {
        Vec4::new(1.0, 0.0, 0.0, 0.0)
    };
    Quat::from_xyzw(q.y, q.z, q.w, q.x).normalize()
}

pub fn sh_dc_to_rgb(dc: [f32; 3]) -> Vec3 {
    const C0: f32 = 0.282_094_8;
    Vec3::new(0.5 + C0 * dc[0], 0.5 + C0 * dc[1], 0.5 + C0 * dc[2]).clamp(Vec3::ZERO, Vec3::ONE)
}

pub fn sh_to_rgb(dc: [f32; 3], f_rest: &[f32], dir: Vec3, requested_degree: u32) -> Vec3 {
    const C0: f32 = 0.282_094_8;
    const C1: f32 = 0.488_602_52;
    const C2: [f32; 5] = [
        1.092_548_5,
        -1.092_548_5,
        0.315_391_57,
        -1.092_548_5,
        0.546_274_24,
    ];
    const C3: [f32; 7] = [
        -0.590_043_6,
        2.890_611_4,
        -0.457_045_8,
        0.373_176_34,
        -0.457_045_8,
        1.445_305_7,
        -0.590_043_6,
    ];

    let degree = requested_degree.min(available_sh_degree(f_rest.len()));
    let rest_per_channel = f_rest.len() / 3;
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;
    let mut rgb = [C0 * dc[0], C0 * dc[1], C0 * dc[2]];

    for channel in 0..3 {
        if degree >= 1 {
            rgb[channel] += -C1 * y * sh_rest(f_rest, rest_per_channel, channel, 1)
                + C1 * z * sh_rest(f_rest, rest_per_channel, channel, 2)
                - C1 * x * sh_rest(f_rest, rest_per_channel, channel, 3);
        }

        if degree >= 2 {
            rgb[channel] += C2[0] * x * y * sh_rest(f_rest, rest_per_channel, channel, 4)
                + C2[1] * y * z * sh_rest(f_rest, rest_per_channel, channel, 5)
                + C2[2]
                    * (2.0 * z * z - x * x - y * y)
                    * sh_rest(f_rest, rest_per_channel, channel, 6)
                + C2[3] * x * z * sh_rest(f_rest, rest_per_channel, channel, 7)
                + C2[4] * (x * x - y * y) * sh_rest(f_rest, rest_per_channel, channel, 8);
        }

        if degree >= 3 {
            rgb[channel] +=
                C3[0] * y * (3.0 * x * x - y * y) * sh_rest(f_rest, rest_per_channel, channel, 9)
                    + C3[1] * x * y * z * sh_rest(f_rest, rest_per_channel, channel, 10)
                    + C3[2]
                        * y
                        * (4.0 * z * z - x * x - y * y)
                        * sh_rest(f_rest, rest_per_channel, channel, 11)
                    + C3[3]
                        * z
                        * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y)
                        * sh_rest(f_rest, rest_per_channel, channel, 12)
                    + C3[4]
                        * x
                        * (4.0 * z * z - x * x - y * y)
                        * sh_rest(f_rest, rest_per_channel, channel, 13)
                    + C3[5] * z * (x * x - y * y) * sh_rest(f_rest, rest_per_channel, channel, 14)
                    + C3[6]
                        * x
                        * (x * x - 3.0 * y * y)
                        * sh_rest(f_rest, rest_per_channel, channel, 15);
        }
    }

    Vec3::new(0.5 + rgb[0], 0.5 + rgb[1], 0.5 + rgb[2]).clamp(Vec3::ZERO, Vec3::ONE)
}

fn available_sh_degree(rest_len: usize) -> u32 {
    match rest_len {
        0..=8 => 0,
        9..=23 => 1,
        24..=44 => 2,
        _ => 3,
    }
}

fn sh_rest(f_rest: &[f32], rest_per_channel: usize, channel: usize, coeff_index: usize) -> f32 {
    if coeff_index == 0 || coeff_index > rest_per_channel {
        return 0.0;
    }
    f_rest
        .get(channel * rest_per_channel + coeff_index - 1)
        .copied()
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_matches_expected_values() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn graphdeco_quaternion_normalizes_wxyz_layout() {
        let q = normalize_graphdeco_quat(Vec4::new(2.0, 0.0, 0.0, 0.0));
        assert!((q.length() - 1.0).abs() < 1e-6);
        assert!((q.w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn depth_sort_orders_back_to_front() {
        let raw = vec![
            sample_raw(Vec3::new(0.0, 0.0, 1.0)),
            sample_raw(Vec3::new(0.0, 0.0, 5.0)),
            sample_raw(Vec3::new(0.0, 0.0, 3.0)),
        ];
        let scene = SplatScene::from_raw(raw, "test".into());
        let sorted = scene.sorted_gpu_from_eye(Vec3::ZERO);
        let depths: Vec<f32> = sorted.iter().map(|s| s.position().z).collect();
        assert_eq!(depths, vec![5.0, 3.0, 1.0]);
    }

    #[test]
    fn camera_depth_sort_culls_behind_camera() {
        let raw = vec![
            sample_raw(Vec3::new(0.0, 0.0, -1.0)),
            sample_raw(Vec3::new(0.0, 0.0, -5.0)),
            sample_raw(Vec3::new(0.0, 0.0, 1.0)),
        ];
        let scene = SplatScene::from_raw(raw, "test".into());
        let view = Mat4::look_at_rh(Vec3::ZERO, -Vec3::Z, Vec3::Y);
        let view_proj = Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.01, 100.0) * view;
        let sorted = scene.sorted_gpu_for_camera(
            view,
            view_proj,
            Vec3::ZERO,
            0,
            0.01,
            100.0,
            DepthSort::BackToFront,
        );
        let depths: Vec<f32> = sorted.iter().map(|s| s.position().z).collect();
        assert_eq!(depths, vec![-5.0, -1.0]);
    }

    #[test]
    fn camera_depth_sort_supports_front_to_back_order() {
        let raw = vec![
            sample_raw(Vec3::new(0.0, 0.0, -1.0)),
            sample_raw(Vec3::new(0.0, 0.0, -5.0)),
            sample_raw(Vec3::new(0.0, 0.0, -3.0)),
        ];
        let scene = SplatScene::from_raw(raw, "test".into());
        let view = Mat4::look_at_rh(Vec3::ZERO, -Vec3::Z, Vec3::Y);
        let view_proj = Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.01, 100.0) * view;
        let sorted = scene.sorted_gpu_for_camera(
            view,
            view_proj,
            Vec3::ZERO,
            0,
            0.01,
            100.0,
            DepthSort::FrontToBack,
        );
        let depths: Vec<f32> = sorted.iter().map(|s| s.position().z).collect();
        assert_eq!(depths, vec![-1.0, -3.0, -5.0]);
    }

    #[test]
    fn camera_depth_sort_culls_far_offscreen_centers() {
        let raw = vec![
            sample_raw(Vec3::new(0.0, 0.0, -3.0)),
            sample_raw(Vec3::new(100.0, 0.0, -3.0)),
        ];
        let scene = SplatScene::from_raw(raw, "test".into());
        let view = Mat4::look_at_rh(Vec3::ZERO, -Vec3::Z, Vec3::Y);
        let view_proj = Mat4::perspective_rh(60.0_f32.to_radians(), 1.0, 0.01, 100.0) * view;
        let sorted = scene.sorted_gpu_for_camera(
            view,
            view_proj,
            Vec3::ZERO,
            0,
            0.01,
            100.0,
            DepthSort::BackToFront,
        );

        assert_eq!(sorted.len(), 1);
        assert_eq!(sorted[0].position(), Vec3::new(0.0, 0.0, -3.0));
    }

    #[test]
    fn detected_sh_degree_uses_rest_coefficient_count() {
        let mut raw = sample_raw(Vec3::ZERO);
        raw.f_rest = vec![0.0; 24];
        let scene = SplatScene::from_raw(vec![raw], "test".into());
        assert_eq!(scene.detected_sh_degree(), 2);
    }

    #[test]
    fn robust_view_bounds_ignore_extreme_outliers() {
        let mut raw = (0..100)
            .map(|index| sample_raw(Vec3::splat(index as f32)))
            .collect::<Vec<_>>();
        raw.push(sample_raw(Vec3::splat(10_000.0)));

        let scene = SplatScene::from_raw(raw, "test".into());

        assert!(scene.radius > 5_000.0);
        assert!(scene.view_radius < 100.0);
    }

    #[test]
    fn sh_degree_zero_matches_dc_color() {
        let dc = [0.25, -0.5, 1.0];
        assert_eq!(sh_to_rgb(dc, &[1.0; 45], Vec3::X, 0), sh_dc_to_rgb(dc));
    }

    #[test]
    fn sh_degree_one_depends_on_view_direction() {
        let mut f_rest = vec![0.0; 9];
        f_rest[2] = 1.0;

        let positive_x = sh_to_rgb([0.0; 3], &f_rest, Vec3::X, 1).x;
        let negative_x = sh_to_rgb([0.0; 3], &f_rest, -Vec3::X, 1).x;

        assert!(negative_x > positive_x);
    }

    fn sample_raw(position: Vec3) -> GaussianRaw {
        GaussianRaw {
            position,
            f_dc: [0.0, 0.0, 0.0],
            f_rest: Vec::new(),
            opacity_logit: 0.0,
            log_scale: Vec3::splat(-2.0),
            rotation_raw: Vec4::new(1.0, 0.0, 0.0, 0.0),
        }
    }
}
