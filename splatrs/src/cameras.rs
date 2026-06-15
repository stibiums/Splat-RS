use std::{fs::File, path::Path};

use anyhow::{Context, Result};
use glam::Vec3;
use serde::Deserialize;

use crate::scene::SplatScene;

#[derive(Clone, Copy, Debug)]
pub struct CameraPreset {
    pub eye: Vec3,
    pub target: Vec3,
    pub fovy_radians: f32,
}

#[derive(Debug, Deserialize)]
struct CameraJson {
    position: [f32; 3],
    rotation: [[f32; 3]; 3],
    height: f32,
    fy: f32,
}

pub fn load_first_preset_for_model(
    model_path: &Path,
    scene: &SplatScene,
) -> Result<Option<CameraPreset>> {
    let Some(cameras_path) = find_cameras_json(model_path) else {
        return Ok(None);
    };

    let file = File::open(&cameras_path)
        .with_context(|| format!("failed to open {}", cameras_path.display()))?;
    let cameras: Vec<CameraJson> = serde_json::from_reader(file)
        .with_context(|| format!("failed to parse {}", cameras_path.display()))?;
    let Some(camera) = cameras.first() else {
        return Ok(None);
    };

    Ok(Some(camera.to_preset(scene)))
}

fn find_cameras_json(model_path: &Path) -> Option<std::path::PathBuf> {
    for ancestor in model_path.ancestors().skip(1) {
        let candidate = ancestor.join("cameras.json");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

impl CameraJson {
    fn to_preset(&self, scene: &SplatScene) -> CameraPreset {
        let eye = Vec3::from_array(self.position);
        let to_scene = (scene.view_center - eye).normalize_or_zero();
        let forward = self
            .forward()
            .map(|dir| if dir.dot(to_scene) >= 0.0 { dir } else { -dir })
            .unwrap_or(to_scene);
        let distance = scene.view_center.distance(eye).max(scene.view_radius);
        let target = eye + forward * distance;
        let fovy_radians = 2.0 * (self.height / (2.0 * self.fy)).atan();

        CameraPreset {
            eye,
            target,
            fovy_radians,
        }
    }

    fn forward(&self) -> Option<Vec3> {
        let c2w_z = Vec3::new(
            self.rotation[0][2],
            self.rotation[1][2],
            self.rotation[2][2],
        );
        let forward = c2w_z.normalize_or_zero();
        if forward.length_squared() > 0.0 {
            Some(forward)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use super::*;
    use crate::scene::GaussianRaw;
    use glam::Vec4;

    #[test]
    fn finds_cameras_json_in_model_ancestor() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("train/point_cloud/iteration_7000");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(dir.path().join("train/cameras.json"), "[]").unwrap();

        let found = find_cameras_json(&model_dir.join("point_cloud.ply")).unwrap();
        assert_eq!(found, dir.path().join("train/cameras.json"));
    }

    #[test]
    fn loads_first_camera_preset() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("train/point_cloud/iteration_7000");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(
            dir.path().join("train/cameras.json"),
            r#"[{"position":[0.0,0.0,-4.0],"rotation":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],"height":1000.0,"fy":1000.0}]"#,
        )
        .unwrap();
        let scene = SplatScene::from_raw(vec![sample_raw(Vec3::ZERO)], "test".into());

        let preset =
            load_first_preset_for_model(Path::new(&model_dir.join("point_cloud.ply")), &scene)
                .unwrap()
                .unwrap();

        assert_eq!(preset.eye, Vec3::new(0.0, 0.0, -4.0));
        assert!(preset.target.z > preset.eye.z);
        assert!((preset.fovy_radians - 0.927_295_2).abs() < 1e-5);
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
