use glam::{Quat, Vec3, Vec4};

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

        Self {
            raw,
            gpu,
            bounds_min,
            bounds_max,
            center,
            radius,
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
        match first.f_rest.len() {
            0 => 0,
            9 => 1,
            24 => 2,
            _ => 3,
        }
    }

    pub fn sorted_gpu_from_eye(&self, eye: Vec3) -> Vec<GaussianGpu> {
        let mut order: Vec<usize> = (0..self.gpu.len()).collect();
        order.sort_unstable_by(|&a, &b| {
            let da = self.gpu[a].position().distance_squared(eye);
            let db = self.gpu[b].position().distance_squared(eye);
            db.total_cmp(&da)
        });
        order.into_iter().map(|index| self.gpu[index]).collect()
    }
}

fn bounds_for(raw: &[GaussianRaw]) -> (Vec3, Vec3) {
    raw.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(min, max), splat| (min.min(splat.position), max.max(splat.position)),
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
    fn detected_sh_degree_uses_rest_coefficient_count() {
        let mut raw = sample_raw(Vec3::ZERO);
        raw.f_rest = vec![0.0; 24];
        let scene = SplatScene::from_raw(vec![raw], "test".into());
        assert_eq!(scene.detected_sh_degree(), 2);
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
