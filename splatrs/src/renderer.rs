use std::{num::NonZeroU64, time::Instant};

use anyhow::{Context, Result};
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};

use crate::{
    camera::Camera,
    scene::{DepthSort, GaussianGpu, SplatScene},
};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Uniforms {
    view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    viewport: [f32; 4],
    focal: [f32; 4],
    options: [f32; 4],
    post: [f32; 4],
    quality: [f32; 4],
    alpha: [f32; 4],
    color: [f32; 4],
}

impl Uniforms {
    pub(crate) fn new(camera: &Camera, size: PhysicalSize<u32>, options: RenderOptions) -> Self {
        let tan_fovy = (camera.fovy_radians * 0.5).tan();
        let tan_fovx = tan_fovy * camera.aspect.max(0.001);
        let focal_x = size.width as f32 / (2.0 * tan_fovx.max(0.001));
        let focal_y = size.height as f32 / (2.0 * tan_fovy.max(0.001));

        Self {
            view_proj: camera.view_projection().to_cols_array_2d(),
            view: camera.view().to_cols_array_2d(),
            viewport: [size.width as f32, size.height as f32, 0.0, 0.0],
            focal: [focal_x, focal_y, tan_fovx, tan_fovy],
            options: [
                options.opacity_scale,
                if options.point_mode { 1.0 } else { 0.0 },
                options.splat_scale,
                options.max_splat_radius,
            ],
            post: [
                options.exposure,
                options.tone_map.shader_value(),
                if options.lowpass_alpha_compensation {
                    1.0
                } else {
                    0.0
                },
                0.0,
            ],
            quality: [
                options.footprint.shader_value(),
                options.lowpass_pixels,
                options.kernel_cutoff,
                options.radius_alpha.shader_value(),
            ],
            alpha: [options.alpha_cutoff, options.max_alpha, 0.0, 0.0],
            color: [options.color_max, options.saturation, 0.0, 0.0],
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Footprint {
    #[default]
    Axes,
    Covariance,
}

impl Footprint {
    fn shader_value(self) -> f32 {
        match self {
            Self::Axes => 0.0,
            Self::Covariance => 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RadiusAlpha {
    #[default]
    Area,
    Linear,
    Preserve,
}

impl RadiusAlpha {
    fn shader_value(self) -> f32 {
        match self {
            Self::Area => 0.0,
            Self::Linear => 1.0,
            Self::Preserve => 2.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CpuSortMode {
    Global,
    #[default]
    TileLocal,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ToneMap {
    #[default]
    None,
    Reinhard,
    Aces,
}

impl ToneMap {
    fn shader_value(self) -> f32 {
        match self {
            Self::None => 0.0,
            Self::Reinhard => 1.0,
            Self::Aces => 2.0,
        }
    }

    pub fn apply(self, color: [f32; 3], exposure: f32) -> [f32; 3] {
        let exposed = [
            (color[0] * exposure).max(0.0),
            (color[1] * exposure).max(0.0),
            (color[2] * exposure).max(0.0),
        ];
        match self {
            Self::None => exposed,
            Self::Reinhard => [
                exposed[0] / (1.0 + exposed[0]),
                exposed[1] / (1.0 + exposed[1]),
                exposed[2] / (1.0 + exposed[2]),
            ],
            Self::Aces => [
                aces_tonemap(exposed[0]),
                aces_tonemap(exposed[1]),
                aces_tonemap(exposed[2]),
            ],
        }
    }

    pub fn apply_color(
        self,
        color: [f32; 3],
        exposure: f32,
        color_max: f32,
        saturation: f32,
    ) -> [f32; 3] {
        let color_max = color_max.max(0.001);
        let saturation = saturation.clamp(0.0, 2.0);
        let clamped = [
            color[0].clamp(0.0, color_max),
            color[1].clamp(0.0, color_max),
            color[2].clamp(0.0, color_max),
        ];
        let luma = clamped[0] * 0.2126 + clamped[1] * 0.7152 + clamped[2] * 0.0722;
        self.apply(
            [
                luma + (clamped[0] - luma) * saturation,
                luma + (clamped[1] - luma) * saturation,
                luma + (clamped[2] - luma) * saturation,
            ],
            exposure,
        )
    }
}

fn aces_tonemap(value: f32) -> f32 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    ((value * (a * value + b)) / (value * (c * value + d) + e)).clamp(0.0, 1.0)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderOptions {
    pub point_mode: bool,
    pub opacity_scale: f32,
    pub splat_scale: f32,
    pub sh_degree: u32,
    pub max_splat_radius: f32,
    pub kernel_cutoff: f32,
    pub lowpass_pixels: f32,
    pub alpha_cutoff: f32,
    pub max_alpha: f32,
    pub color_max: f32,
    pub saturation: f32,
    pub footprint: Footprint,
    pub radius_alpha: RadiusAlpha,
    pub background: [f32; 3],
    pub exposure: f32,
    pub tone_map: ToneMap,
    pub lowpass_alpha_compensation: bool,
    pub cpu_sort_mode: CpuSortMode,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            point_mode: false,
            opacity_scale: 1.5,
            splat_scale: 0.4,
            sh_degree: 0,
            max_splat_radius: 80.0,
            kernel_cutoff: 8.0,
            lowpass_pixels: 0.3,
            alpha_cutoff: 0.006,
            max_alpha: 0.99,
            color_max: 2.0,
            saturation: 1.05,
            footprint: Footprint::Axes,
            radius_alpha: RadiusAlpha::Area,
            background: [0.72, 0.80, 0.92],
            exposure: 0.9,
            tone_map: ToneMap::None,
            lowpass_alpha_compensation: false,
            cpu_sort_mode: CpuSortMode::TileLocal,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SortRequest {
    None,
    Throttled,
    Immediate,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RenderStats {
    pub sorted: bool,
    pub drawn_instances: usize,
    pub full_instances: usize,
}

pub struct Renderer<'window> {
    surface: wgpu::Surface<'window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer,
    instance_capacity: usize,
    interactive_instance_buffer: wgpu::Buffer,
    interactive_instance_capacity: usize,
    interactive_max_splats: usize,
    last_sort: Instant,
    sort_interval: std::time::Duration,
    sorted_instances: Vec<GaussianGpu>,
    interactive_instances: Vec<GaussianGpu>,
}

impl<'window> Renderer<'window> {
    pub async fn new(
        window: &'window Window,
        scene: &SplatScene,
        camera: &Camera,
        sort_interval: std::time::Duration,
        interactive_max_splats: usize,
    ) -> Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window)
            .context("failed to create wgpu surface")?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("failed to request a suitable GPU adapter")?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("splatrs-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .context("failed to create wgpu device")?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|format| format.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let present_mode = surface_caps
            .present_modes
            .iter()
            .copied()
            .find(|mode| *mode == wgpu::PresentMode::Mailbox)
            .unwrap_or(wgpu::PresentMode::Fifo);
        let alpha_mode = surface_caps
            .alpha_modes
            .iter()
            .copied()
            .find(|mode| *mode == wgpu::CompositeAlphaMode::Opaque)
            .unwrap_or(surface_caps.alpha_modes[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let uniforms = Uniforms::new(camera, size, RenderOptions::default());
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniform-buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_layout = create_uniform_layout(&device);
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform-bind-group"),
            layout: &uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline = create_splat_pipeline(&device, surface_format, &uniform_layout);

        let sorted_instances = scene.sorted_gpu_for_camera(
            camera.view(),
            camera.view_projection(),
            camera.eye(),
            0,
            camera.z_near,
            camera.z_far,
            DepthSort::BackToFront,
        );
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instance-buffer"),
            contents: bytemuck::cast_slice(&sorted_instances),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let interactive_instances =
            decimate_preserving_order(&sorted_instances, interactive_max_splats);
        let interactive_instance_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("interactive-instance-buffer"),
                contents: bytemuck::cast_slice(&interactive_instances),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            pipeline,
            uniform_buffer,
            uniform_bind_group,
            instance_buffer,
            instance_capacity: sorted_instances.len(),
            interactive_instance_buffer,
            interactive_instance_capacity: interactive_instances.len(),
            interactive_max_splats,
            last_sort: Instant::now(),
            sort_interval,
            sorted_instances,
            interactive_instances,
        })
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn output_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    pub fn render(
        &mut self,
        scene: &SplatScene,
        camera: &Camera,
        options: RenderOptions,
        sort_request: SortRequest,
    ) -> Result<RenderStats> {
        self.render_with_overlay(scene, camera, options, sort_request, |_, _, _, _, _| {})
    }

    pub fn render_with_overlay<F>(
        &mut self,
        scene: &SplatScene,
        camera: &Camera,
        options: RenderOptions,
        sort_request: SortRequest,
        overlay: F,
    ) -> Result<RenderStats>
    where
        F: FnOnce(
            &wgpu::Device,
            &wgpu::Queue,
            &mut wgpu::CommandEncoder,
            &wgpu::TextureView,
            PhysicalSize<u32>,
        ),
    {
        let uniforms = Uniforms::new(camera, self.size, options);
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let sort_due =
            self.sort_interval.is_zero() || self.last_sort.elapsed() >= self.sort_interval;
        let should_sort = match sort_request {
            SortRequest::None => false,
            SortRequest::Throttled => sort_due,
            SortRequest::Immediate => true,
        };
        let mut sorted = false;
        if should_sort {
            self.sorted_instances = scene.sorted_gpu_for_camera(
                camera.view(),
                camera.view_projection(),
                camera.eye(),
                options.sh_degree,
                camera.z_near,
                camera.z_far,
                DepthSort::BackToFront,
            );
            self.last_sort = Instant::now();
            self.upload_instances();
            self.update_interactive_instances();
            sorted = true;
        }

        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                return Ok(self.stats(sorted, sort_request));
            }
            Err(wgpu::SurfaceError::Timeout) => return Ok(self.stats(sorted, sort_request)),
            Err(wgpu::SurfaceError::OutOfMemory) => anyhow::bail!("wgpu surface ran out of memory"),
        };
        let use_interactive_lod = sort_request == SortRequest::Throttled
            && self.interactive_instances.len() < self.sorted_instances.len();
        let draw_instances = if use_interactive_lod {
            &self.interactive_instances
        } else {
            &self.sorted_instances
        };
        let draw_buffer = if use_interactive_lod {
            &self.interactive_instance_buffer
        } else {
            &self.instance_buffer
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let background = options.tone_map.apply(options.background, options.exposure);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render-encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("splat-render-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: background[0] as f64,
                            g: background[1] as f64,
                            b: background[2] as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, draw_buffer.slice(..));
            pass.draw(0..4, 0..draw_instances.len() as u32);
        }
        overlay(&self.device, &self.queue, &mut encoder, &view, self.size);
        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(RenderStats {
            sorted,
            drawn_instances: draw_instances.len(),
            full_instances: self.sorted_instances.len(),
        })
    }

    fn upload_instances(&mut self) {
        if self.sorted_instances.len() > self.instance_capacity {
            self.instance_capacity = self.sorted_instances.len();
            self.instance_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("instance-buffer"),
                        contents: bytemuck::cast_slice(&self.sorted_instances),
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    });
        } else {
            self.queue.write_buffer(
                &self.instance_buffer,
                0,
                bytemuck::cast_slice(&self.sorted_instances),
            );
        }
    }

    fn update_interactive_instances(&mut self) {
        self.interactive_instances =
            decimate_preserving_order(&self.sorted_instances, self.interactive_max_splats);
        if self.interactive_instances.len() > self.interactive_instance_capacity {
            self.interactive_instance_capacity = self.interactive_instances.len();
            self.interactive_instance_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("interactive-instance-buffer"),
                        contents: bytemuck::cast_slice(&self.interactive_instances),
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    });
        } else {
            self.queue.write_buffer(
                &self.interactive_instance_buffer,
                0,
                bytemuck::cast_slice(&self.interactive_instances),
            );
        }
    }

    fn stats(&self, sorted: bool, sort_request: SortRequest) -> RenderStats {
        let use_interactive_lod = sort_request == SortRequest::Throttled
            && self.interactive_instances.len() < self.sorted_instances.len();
        RenderStats {
            sorted,
            drawn_instances: if use_interactive_lod {
                self.interactive_instances.len()
            } else {
                self.sorted_instances.len()
            },
            full_instances: self.sorted_instances.len(),
        }
    }
}

fn decimate_preserving_order(items: &[GaussianGpu], max_items: usize) -> Vec<GaussianGpu> {
    if max_items == 0 || max_items >= items.len() {
        return items.to_vec();
    }

    let len = items.len();
    let mut selected = Vec::with_capacity(max_items);
    for sample_index in 0..max_items {
        let index = ((sample_index * 2 + 1) * len) / (max_items * 2);
        selected.push(items[index.min(len - 1)]);
    }
    selected
}

pub(crate) fn create_uniform_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("uniform-layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(std::mem::size_of::<Uniforms>() as u64),
            },
            count: None,
        }],
    })
}

pub(crate) fn create_splat_pipeline(
    device: &wgpu::Device,
    color_format: wgpu::TextureFormat,
    uniform_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("splat-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline-layout"),
        bind_group_layouts: &[uniform_layout],
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("splat-pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[instance_layout()],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        multiview: None,
        cache: None,
    })
}

pub(crate) fn instance_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
        0 => Float32x4,
        1 => Float32x4,
        2 => Float32x4,
        3 => Float32x4,
        4 => Float32x4
    ];
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<GaussianGpu>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &ATTRIBUTES,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_parses_as_wgsl() {
        naga::front::wgsl::parse_str(include_str!("shader.wgsl")).unwrap();
    }

    #[test]
    fn decimation_preserves_order_and_limit() {
        let items = (0..10).map(sample_gpu).collect::<Vec<_>>();

        let decimated = decimate_preserving_order(&items, 4);
        let positions = decimated
            .iter()
            .map(|splat| splat.position_opacity[0] as i32)
            .collect::<Vec<_>>();

        assert_eq!(positions, vec![1, 3, 6, 8]);
    }

    #[test]
    fn zero_decimation_limit_keeps_full_quality() {
        let items = (0..5).map(sample_gpu).collect::<Vec<_>>();

        let decimated = decimate_preserving_order(&items, 0);

        assert_eq!(decimated.len(), items.len());
    }

    fn sample_gpu(index: i32) -> GaussianGpu {
        GaussianGpu {
            position_opacity: [index as f32, 0.0, 0.0, 1.0],
            color: [1.0, 1.0, 1.0, 1.0],
            axis0_radius: [1.0, 0.0, 0.0, 1.0],
            axis1_radius: [0.0, 1.0, 0.0, 1.0],
            axis2_radius: [0.0, 0.0, 1.0, 1.0],
        }
    }
}
