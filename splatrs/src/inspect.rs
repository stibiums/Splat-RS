use std::mem::size_of;

use anyhow::Result;

use crate::{
    cli::InspectArgs,
    loader,
    scene::{GaussianGpu, GaussianRaw},
};

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

    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_byte_counts() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KiB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MiB");
    }
}
