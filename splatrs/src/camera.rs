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
        }
    }

    pub fn from_eye_target(
        eye: Vec3,
        target: Vec3,
        radius: f32,
        aspect: f32,
        fovy_radians: f32,
    ) -> Self {
        let offset = eye - target;
        let distance = offset.length().max(0.01);
        let direction = offset / distance;
        Self {
            target,
            yaw: direction.x.atan2(direction.z),
            pitch: direction.y.asin().clamp(-1.45, 1.45),
            distance,
            aspect,
            fovy_radians,
            z_near: (radius * 0.001).max(0.001),
            z_far: (radius * 20.0).max(100.0),
        }
    }

    pub fn eye(&self) -> Vec3 {
        let cos_pitch = self.pitch.cos();
        let direction = Vec3::new(
            self.yaw.sin() * cos_pitch,
            self.pitch.sin(),
            self.yaw.cos() * cos_pitch,
        );
        self.target + direction * self.distance
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
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
        self.pitch = (self.pitch - delta_y * 0.006).clamp(-1.45, 1.45);
    }

    pub fn zoom(&mut self, scroll_delta: f32) {
        let scale = (1.0 - scroll_delta * 0.08).clamp(0.2, 5.0);
        self.distance = (self.distance * scale).max(0.01);
    }
}
