use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    sync::mpsc,
};

use anyhow::{Context, Result};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalSize;

use crate::{
    camera::Camera,
    cameras,
    cli::{ContactSheetArgs, RenderArgs},
    loader,
    renderer::{RenderOptions, Uniforms, create_splat_pipeline, create_uniform_layout},
    scene::{DepthSort, SplatScene},
};

const BACKGROUND: [f32; 3] = [0.015, 0.017, 0.02];
const LABEL_HEIGHT: u32 = 24;

pub fn run(args: RenderArgs) -> Result<()> {
    let scene = loader::load_scene(&args.model, args.max_splats)?;
    let width = args.width.max(1);
    let height = args.height.max(1);
    let camera = make_camera(&args, &scene, width, height);
    let options = RenderOptions {
        point_mode: false,
        opacity_scale: args.opacity_scale.clamp(0.05, 8.0),
        splat_scale: args.splat_scale.clamp(0.05, 12.0),
        sh_degree: args.sh_degree.as_u32(),
        max_splat_radius: args.max_splat_radius.clamp(2.0, 1024.0),
    };

    let pixels = pollster::block_on(render_offscreen(&scene, &camera, options, width, height))?;
    write_bmp(&args.output, width, height, &pixels)?;

    tracing::info!(
        "rendered {} splats to {}",
        scene.len(),
        args.output.display()
    );
    Ok(())
}

pub fn run_contact_sheet(args: ContactSheetArgs) -> Result<()> {
    let scene = loader::load_scene(&args.model, args.max_splats)?;
    let tile_width = args.width.max(1);
    let tile_height = args.height.max(1);
    let columns = args.columns.max(1).min(args.camera_indices.len().max(1));
    let rows = args.camera_indices.len().max(1).div_ceil(columns);
    let sheet_width = tile_width
        .checked_mul(columns as u32)
        .context("contact sheet width overflowed")?;
    let tile_stride_y = tile_height
        .checked_add(LABEL_HEIGHT)
        .context("contact sheet tile height overflowed")?;
    let sheet_height = tile_stride_y
        .checked_mul(rows as u32)
        .context("contact sheet height overflowed")?;
    let mut sheet = vec![0; sheet_width as usize * sheet_height as usize * 4];
    fill_rgba(&mut sheet, [18, 20, 24, 255]);

    let options = RenderOptions {
        point_mode: false,
        opacity_scale: args.opacity_scale.clamp(0.05, 8.0),
        splat_scale: args.splat_scale.clamp(0.05, 12.0),
        sh_degree: args.sh_degree.as_u32(),
        max_splat_radius: args.max_splat_radius.clamp(2.0, 1024.0),
    };

    for (tile_index, &camera_index) in args.camera_indices.iter().enumerate() {
        let camera = make_camera_for(&args.model, &scene, camera_index, tile_width, tile_height);
        let pixels = pollster::block_on(render_offscreen(
            &scene,
            &camera,
            options,
            tile_width,
            tile_height,
        ))
        .with_context(|| format!("failed to render camera {camera_index}"))?;
        let tile_x = (tile_index % columns) as u32 * tile_width;
        let tile_y = (tile_index / columns) as u32 * tile_stride_y;
        draw_label(
            &mut sheet,
            sheet_width,
            sheet_height,
            tile_x + 8,
            tile_y + 6,
            camera_index,
        );
        blit_rgba(
            &mut sheet,
            sheet_width,
            sheet_height,
            &pixels,
            tile_width,
            tile_height,
            tile_x,
            tile_y + LABEL_HEIGHT,
        );
    }

    write_bmp(&args.output, sheet_width, sheet_height, &sheet)?;
    tracing::info!(
        "rendered {} camera tiles from {} splats to {}",
        args.camera_indices.len(),
        scene.len(),
        args.output.display()
    );
    Ok(())
}

async fn render_offscreen(
    scene: &SplatScene,
    camera: &Camera,
    options: RenderOptions,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .context("failed to request a suitable GPU adapter")?;
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("splatrs-headless-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .context("failed to create wgpu device")?;

    let format = wgpu::TextureFormat::Rgba8Unorm;
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless-target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let uniforms = Uniforms::new(camera, PhysicalSize::new(width, height), options);
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("headless-uniform-buffer"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let uniform_layout = create_uniform_layout(&device);
    let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("headless-uniform-bind-group"),
        layout: &uniform_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });
    let pipeline = create_splat_pipeline(&device, format, &uniform_layout);

    let sorted_instances = scene.sorted_gpu_for_camera(
        camera.view(),
        camera.view_projection(),
        camera.eye(),
        options.sh_degree,
        camera.z_near,
        camera.z_far,
        DepthSort::FrontToBack,
    );
    let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("headless-instance-buffer"),
        contents: bytemuck::cast_slice(&sorted_instances),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let bytes_per_pixel = 4;
    let unpadded_bytes_per_row = width * bytes_per_pixel;
    let padded_bytes_per_row = align_to(unpadded_bytes_per_row, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let output_buffer_size = padded_bytes_per_row as u64 * height as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("headless-output-buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("headless-render-encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("headless-render-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &uniform_bind_group, &[]);
        pass.set_vertex_buffer(0, instance_buffer.slice(..));
        pass.draw(0..4, 0..sorted_instances.len() as u32);
    }
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &output_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let buffer_slice = output_buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait).panic_on_timeout();
    receiver
        .recv()
        .context("failed to receive GPU readback status")?
        .context("failed to map GPU readback buffer")?;

    let mapped = buffer_slice.get_mapped_range();
    let pixels = composite_readback(
        &mapped,
        width,
        height,
        unpadded_bytes_per_row as usize,
        padded_bytes_per_row as usize,
    );
    drop(mapped);
    output_buffer.unmap();

    Ok(pixels)
}

fn make_camera(args: &RenderArgs, scene: &SplatScene, width: u32, height: u32) -> Camera {
    make_camera_for(&args.model, scene, args.camera_index, width, height)
}

fn make_camera_for(
    model: &Path,
    scene: &SplatScene,
    camera_index: usize,
    width: u32,
    height: u32,
) -> Camera {
    let aspect = width as f32 / height as f32;
    match cameras::load_preset_for_model(model, scene, camera_index) {
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

fn fill_rgba(pixels: &mut [u8], color: [u8; 4]) {
    for pixel in pixels.chunks_exact_mut(4) {
        pixel.copy_from_slice(&color);
    }
}

fn blit_rgba(
    dst: &mut [u8],
    dst_width: u32,
    dst_height: u32,
    src: &[u8],
    src_width: u32,
    src_height: u32,
    dst_x: u32,
    dst_y: u32,
) {
    let copy_width = src_width.min(dst_width.saturating_sub(dst_x)) as usize;
    let copy_height = src_height.min(dst_height.saturating_sub(dst_y)) as usize;
    for y in 0..copy_height {
        let src_start = y * src_width as usize * 4;
        let dst_start = ((dst_y as usize + y) * dst_width as usize + dst_x as usize) * 4;
        let byte_count = copy_width * 4;
        dst[dst_start..dst_start + byte_count]
            .copy_from_slice(&src[src_start..src_start + byte_count]);
    }
}

fn draw_label(pixels: &mut [u8], width: u32, height: u32, x: u32, y: u32, value: usize) {
    let mut cursor = x;
    for digit in value.to_string().bytes() {
        draw_glyph(pixels, width, height, cursor, y, (digit - b'0') as usize);
        cursor += 12;
    }
}

fn draw_glyph(pixels: &mut [u8], width: u32, height: u32, x: u32, y: u32, glyph: usize) {
    const GLYPHS: [[u8; 5]; 10] = [
        [0b111, 0b101, 0b101, 0b101, 0b111],
        [0b010, 0b110, 0b010, 0b010, 0b111],
        [0b111, 0b001, 0b111, 0b100, 0b111],
        [0b111, 0b001, 0b111, 0b001, 0b111],
        [0b101, 0b101, 0b111, 0b001, 0b001],
        [0b111, 0b100, 0b111, 0b001, 0b111],
        [0b111, 0b100, 0b111, 0b101, 0b111],
        [0b111, 0b001, 0b010, 0b010, 0b010],
        [0b111, 0b101, 0b111, 0b101, 0b111],
        [0b111, 0b101, 0b111, 0b001, 0b111],
    ];
    let Some(rows) = GLYPHS.get(glyph) else {
        return;
    };
    for (row, bits) in rows.iter().copied().enumerate() {
        for col in 0..3 {
            if bits & (1 << (2 - col)) == 0 {
                continue;
            }
            draw_block(pixels, width, height, x + col * 3, y + row as u32 * 3);
        }
    }
}

fn draw_block(pixels: &mut [u8], width: u32, height: u32, x: u32, y: u32) {
    for dy in 0..2 {
        for dx in 0..2 {
            let px = x + dx;
            let py = y + dy;
            if px >= width || py >= height {
                continue;
            }
            let index = (py as usize * width as usize + px as usize) * 4;
            pixels[index..index + 4].copy_from_slice(&[235, 238, 242, 255]);
        }
    }
}

fn composite_readback(
    readback: &[u8],
    width: u32,
    height: u32,
    unpadded_bytes_per_row: usize,
    padded_bytes_per_row: usize,
) -> Vec<u8> {
    let mut pixels = vec![0; width as usize * height as usize * 4];
    for y in 0..height as usize {
        let src_row = &readback[y * padded_bytes_per_row..][..unpadded_bytes_per_row];
        let dst_row = &mut pixels[y * unpadded_bytes_per_row..][..unpadded_bytes_per_row];
        for x in 0..width as usize {
            let src = &src_row[x * 4..x * 4 + 4];
            let dst = &mut dst_row[x * 4..x * 4 + 4];
            let alpha = src[3] as f32 / 255.0;
            for channel in 0..3 {
                let premultiplied = src[channel] as f32 / 255.0;
                let linear = (premultiplied + (1.0 - alpha) * BACKGROUND[channel]).clamp(0.0, 1.0);
                dst[channel] = (linear_to_srgb(linear) * 255.0).round() as u8;
            }
            dst[3] = 255;
        }
    }
    pixels
}

fn write_bmp(path: &Path, width: u32, height: u32, rgba: &[u8]) -> Result<()> {
    let row_bytes = width
        .checked_mul(4)
        .context("BMP row byte count overflowed")?;
    let pixel_bytes = row_bytes
        .checked_mul(height)
        .context("BMP pixel byte count overflowed")?;
    let file_size = 54_u32
        .checked_add(pixel_bytes)
        .context("BMP file size overflowed")?;
    let expected_len = pixel_bytes as usize;
    if rgba.len() != expected_len {
        anyhow::bail!(
            "RGBA buffer has {} bytes, expected {expected_len}",
            rgba.len()
        );
    }

    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writer.write_all(b"BM")?;
    writer.write_all(&file_size.to_le_bytes())?;
    writer.write_all(&[0; 4])?;
    writer.write_all(&54_u32.to_le_bytes())?;
    writer.write_all(&40_u32.to_le_bytes())?;
    writer.write_all(&(width as i32).to_le_bytes())?;
    writer.write_all(&(-(height as i32)).to_le_bytes())?;
    writer.write_all(&1_u16.to_le_bytes())?;
    writer.write_all(&32_u16.to_le_bytes())?;
    writer.write_all(&0_u32.to_le_bytes())?;
    writer.write_all(&pixel_bytes.to_le_bytes())?;
    writer.write_all(&2835_i32.to_le_bytes())?;
    writer.write_all(&2835_i32.to_le_bytes())?;
    writer.write_all(&0_u32.to_le_bytes())?;
    writer.write_all(&0_u32.to_le_bytes())?;

    for pixel in rgba.chunks_exact(4) {
        writer.write_all(&[pixel[2], pixel[1], pixel[0], pixel[3]])?;
    }
    writer.flush()?;
    Ok(())
}

fn align_to(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}
