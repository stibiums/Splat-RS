use glam::{Mat4, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub target: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub aspect: f32,
    pub fovy_radians: f32,
    pub z_near: f32,
    pub z_far: f32,
    pub up: Vec3,
    orbit_right: Vec3,
    orbit_up: Vec3,
    orbit_back: Vec3,
}

impl Camera {
    pub fn for_scene(center: Vec3, radius: f32, aspect: f32) -> Self {
        Self {
            target: center,
            yaw: 0.0,
            pitch: 0.25,
            distance: (radius * 2.5).max(0.1),
            aspect,
            fovy_radians: 50.0_f32.to_radians(),
            z_near: (radius * 0.001).max(0.001),
            z_far: (radius * 20.0).max(100.0),
            up: Vec3::Y,
            orbit_right: Vec3::X,
            orbit_up: Vec3::Y,
            orbit_back: Vec3::Z,
        }
    }

    pub fn from_eye_target_up(
        eye: Vec3,
        target: Vec3,
        up: Vec3,
        radius: f32,
        aspect: f32,
        fovy_radians: f32,
    ) -> Self {
        let offset = eye - target;
        let distance = offset.length().max(0.01);
        let back = offset / distance;
        let (right, up, back) = orbit_basis(back, up);
        Self {
            target,
            yaw: 0.0,
            pitch: 0.0,
            distance,
            aspect,
            fovy_radians,
            z_near: (radius * 0.001).max(0.001),
            z_far: (radius * 20.0).max(100.0),
            up,
            orbit_right: right,
            orbit_up: up,
            orbit_back: back,
        }
    }

    pub fn eye(&self) -> Vec3 {
        let cos_pitch = self.pitch.cos();
        let local = Vec3::new(
            self.yaw.sin() * cos_pitch,
            self.pitch.sin(),
            self.yaw.cos() * cos_pitch,
        );
        let direction =
            self.orbit_right * local.x + self.orbit_up * local.y + self.orbit_back * local.z;
        self.target + direction * self.distance
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, self.up)
    }

    pub fn projection(&self) -> Mat4 {
        Mat4::perspective_rh(
            self.fovy_radians,
            self.aspect.max(0.001),
            self.z_near,
            self.z_far,
        )
    }

    pub fn view_projection(&self) -> Mat4 {
        self.projection() * self.view()
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width.max(1) as f32 / height.max(1) as f32;
    }

    pub fn orbit(&mut self, delta_x: f32, delta_y: f32) {
        self.yaw -= delta_x * 0.006;
        self.pitch = (self.pitch + delta_y * 0.006).clamp(-1.45, 1.45);
    }

    pub fn zoom(&mut self, scroll_delta: f32) {
        let scale = (1.0 - scroll_delta * 0.08).clamp(0.2, 5.0);
        self.distance = (self.distance * scale).max(0.01);
    }
}

fn normalize_up(up: Vec3) -> Vec3 {
    let normalized = up.normalize_or_zero();
    if normalized.length_squared() > 0.0 {
        normalized
    } else {
        Vec3::Y
    }
}

fn orbit_basis(back: Vec3, up: Vec3) -> (Vec3, Vec3, Vec3) {
    let back = back.normalize_or_zero();
    let back = if back.length_squared() > 0.0 {
        back
    } else {
        Vec3::Z
    };
    let mut up = normalize_up(up);
    let mut right = up.cross(back).normalize_or_zero();
    if right.length_squared() == 0.0 {
        let fallback_up = if back.dot(Vec3::Y).abs() < 0.95 {
            Vec3::Y
        } else {
            Vec3::X
        };
        right = fallback_up.cross(back).normalize_or_zero();
        up = back.cross(right).normalize_or_zero();
    } else {
        up = back.cross(right).normalize_or_zero();
    }
    (right, up, back)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imported_camera_keeps_exact_initial_eye() {
        let eye = Vec3::new(3.0, -0.2, -1.3);
        let forward = Vec3::new(-0.97, -0.006, 0.24).normalize();
        let target = eye + forward * 12.0;
        let up = Vec3::new(-0.03, 0.995, -0.1).normalize();

        let camera = Camera::from_eye_target_up(eye, target, up, 10.0, 16.0 / 9.0, 0.8);

        assert!(camera.eye().distance(eye) < 1e-5);
    }

    #[test]
    fn imported_camera_orbits_in_local_basis() {
        let eye = Vec3::new(0.0, 0.0, 4.0);
        let target = Vec3::ZERO;
        let up = -Vec3::Y;
        let mut camera = Camera::from_eye_target_up(eye, target, up, 10.0, 1.0, 0.8);

        camera.orbit(100.0, 0.0);

        assert!(camera.eye().y.abs() < 1e-5);
        assert_eq!(camera.up, -Vec3::Y);
    }

    #[test]
    fn positive_vertical_drag_increases_pitch() {
        let mut camera = Camera::from_eye_target_up(
            Vec3::new(0.0, 0.0, 4.0),
            Vec3::ZERO,
            Vec3::Y,
            10.0,
            1.0,
            0.8,
        );

        camera.orbit(0.0, 10.0);

        assert!(camera.eye().y > 0.0);
    }
}
