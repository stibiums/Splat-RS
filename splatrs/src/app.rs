use std::{
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use egui_wgpu::{Renderer as EguiRenderer, ScreenDescriptor};
use egui_winit::State as EguiWinitState;
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalSize, PhysicalPosition},
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes, WindowId},
};

use crate::{
    camera::Camera,
    cameras,
    cli::{RenderBackend, ViewArgs},
    headless, loader,
    renderer::{RenderOptions, Renderer, SortRequest, ToneMap},
    scene::SplatScene,
};

pub fn run(args: ViewArgs) -> Result<()> {
    let scene = loader::load_scene(&args.model, args.filters.load_options(args.max_splats))?;
    let sh_degree = args.sh_degree.resolve(scene.detected_sh_degree());
    tracing::info!(
        "loaded {} splats from {} (bounds {:?} .. {:?}, file SH degree {}, active SH degree {})",
        scene.len(),
        scene.source_label,
        scene.bounds_min,
        scene.bounds_max,
        scene.detected_sh_degree(),
        sh_degree
    );

    let event_loop = EventLoop::new()?;
    let mut app = ViewerApp::new(args, scene);
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct ViewerApp<'window> {
    args: ViewArgs,
    scene: SplatScene,
    window: Option<&'static Window>,
    window_id: Option<WindowId>,
    renderer: Option<Renderer<'window>>,
    camera: Option<Camera>,
    initial_camera: Option<Camera>,
    render_options: RenderOptions,
    initial_render_options: RenderOptions,
    egui_ctx: egui::Context,
    egui_state: Option<EguiWinitState>,
    egui_renderer: Option<EguiRenderer>,
    status_message: Option<String>,
    dragging: bool,
    last_cursor: Option<PhysicalPosition<f64>>,
    sort_request: SortRequest,
    last_drawn_instances: usize,
    frame_counter: FrameCounter,
}

impl<'window> ViewerApp<'window> {
    fn new(args: ViewArgs, scene: SplatScene) -> Self {
        let initial_splat_count = scene.len();
        let render_options = Self::render_options_from_args(&args, &scene);

        Self {
            args,
            scene,
            window: None,
            window_id: None,
            renderer: None,
            camera: None,
            initial_camera: None,
            render_options,
            initial_render_options: render_options,
            egui_ctx: egui::Context::default(),
            egui_state: None,
            egui_renderer: None,
            status_message: None,
            dragging: false,
            last_cursor: None,
            sort_request: SortRequest::Immediate,
            last_drawn_instances: initial_splat_count,
            frame_counter: FrameCounter::default(),
        }
    }

    fn render(&mut self) {
        let Some(window) = self.window else {
            return;
        };
        let egui_frame = self.prepare_egui_frame(window);
        let Some(camera) = self.camera.as_ref() else {
            return;
        };
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        let render_result = if let (Some(egui_renderer), Some(egui_frame)) =
            (self.egui_renderer.as_mut(), egui_frame)
        {
            let textures_delta = egui_frame.textures_delta;
            let paint_jobs = egui_frame.paint_jobs;
            let screen_descriptor = egui_frame.screen_descriptor;
            let free_textures = textures_delta.free.clone();
            let result = renderer.render_with_overlay(
                &self.scene,
                camera,
                self.render_options,
                self.sort_request,
                |device, queue, encoder, view, _size| {
                    for (id, image_delta) in &textures_delta.set {
                        egui_renderer.update_texture(device, queue, *id, image_delta);
                    }
                    let user_cmd_bufs = egui_renderer.update_buffers(
                        device,
                        queue,
                        encoder,
                        &paint_jobs,
                        &screen_descriptor,
                    );
                    debug_assert!(
                        user_cmd_bufs.is_empty(),
                        "SplatRS UI does not use egui paint callbacks"
                    );
                    let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("egui-render-pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    egui_renderer.render(
                        &mut pass.forget_lifetime(),
                        &paint_jobs,
                        &screen_descriptor,
                    );
                },
            );
            for id in &free_textures {
                egui_renderer.free_texture(id);
            }
            result
        } else {
            renderer.render(&self.scene, camera, self.render_options, self.sort_request)
        };

        match render_result {
            Ok(stats) => {
                self.last_drawn_instances = stats.drawn_instances;
                if stats.sorted || self.sort_request == SortRequest::Immediate {
                    self.sort_request = SortRequest::None;
                }
            }
            Err(error) => {
                tracing::error!("{error:#}");
                window.set_title(&format!("SplatRS - render error: {error}"));
            }
        }

        if let Some(fps) = self.frame_counter.tick() {
            window.set_title(&format!(
                "SplatRS - {}/{} splats - {:.1} FPS - {} - opacity {:.2} - scale {:.2} - radius {:.0}px - exposure {:.2} - {:?} - SH d{} - {}",
                self.last_drawn_instances,
                self.scene.len(),
                fps,
                if self.render_options.point_mode {
                    "points"
                } else {
                    "splats"
                },
                self.render_options.opacity_scale,
                self.render_options.splat_scale,
                self.render_options.max_splat_radius,
                self.render_options.exposure,
                self.render_options.tone_map,
                self.render_options.sh_degree,
                self.scene.source_label,
            ));
        }
    }
}

struct EguiFrame {
    textures_delta: egui::TexturesDelta,
    paint_jobs: Vec<egui::ClippedPrimitive>,
    screen_descriptor: ScreenDescriptor,
}

#[derive(Default)]
struct UiActions {
    open_scene: bool,
    apply_camera: bool,
    screenshot: bool,
    reset_camera: bool,
    request_sort: bool,
}

impl<'window> ApplicationHandler for ViewerApp<'window> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attributes = WindowAttributes::default()
            .with_title("SplatRS")
            .with_inner_size(LogicalSize::new(self.args.width, self.args.height));
        let window = event_loop
            .create_window(attributes)
            .expect("failed to create window");
        let window_id = window.id();
        let window: &'static Window = Box::leak(Box::new(window));
        let size = window.inner_size();
        let camera = self.make_initial_camera(size.width, size.height);
        let renderer = pollster::block_on(Renderer::new(
            window,
            &self.scene,
            &camera,
            Duration::from_millis(self.args.sort_interval_ms),
            self.args.interactive_max_splats,
        ))
        .expect("failed to initialize renderer");
        let egui_state = EguiWinitState::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(renderer.device().limits().max_texture_dimension_2d as usize),
        );
        let egui_renderer =
            EguiRenderer::new(renderer.device(), renderer.output_format(), None, 1, false);

        self.window = Some(window);
        self.window_id = Some(window_id);
        self.initial_camera = Some(camera);
        self.camera = Some(camera);
        self.renderer = Some(renderer);
        self.egui_state = Some(egui_state);
        self.egui_renderer = Some(egui_renderer);
        self.sort_request = SortRequest::Immediate;
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if Some(window_id) != self.window_id {
            return;
        }

        let egui_response = self
            .window
            .zip(self.egui_state.as_mut())
            .map(|(window, egui_state)| egui_state.on_window_event(window, &event));
        let egui_consumed = egui_response
            .as_ref()
            .is_some_and(|response| response.consumed);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => self.render(),
            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(size);
                }
                if let Some(camera) = self.camera.as_mut() {
                    camera.resize(size.width, size.height);
                }
                self.sort_request = SortRequest::Immediate;
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                    PhysicalKey::Code(KeyCode::KeyR) if !egui_consumed => {
                        if let Some(window) = self.window {
                            let size = window.inner_size();
                            let next_camera = self.initial_camera.unwrap_or_else(|| {
                                self.make_initial_camera(size.width, size.height)
                            });
                            if let Some(camera) = self.camera.as_mut() {
                                *camera = next_camera;
                            }
                            self.sort_request = SortRequest::Immediate;
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                if egui_consumed {
                    self.dragging = false;
                    self.last_cursor = None;
                } else {
                    let was_dragging = self.dragging;
                    self.dragging = state == ElementState::Pressed;
                    if !self.dragging {
                        self.last_cursor = None;
                        if was_dragging {
                            self.sort_request = SortRequest::Immediate;
                        }
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.dragging && !egui_consumed {
                    if let (Some(previous), Some(camera)) = (self.last_cursor, self.camera.as_mut())
                    {
                        camera.orbit(
                            (position.x - previous.x) as f32,
                            (position.y - previous.y) as f32,
                        );
                        self.sort_request = SortRequest::Throttled;
                    }
                    self.last_cursor = Some(position);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if !egui_consumed {
                    let scroll = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(position) => (position.y as f32) / 40.0,
                    };
                    if let Some(camera) = self.camera.as_mut() {
                        camera.zoom(scroll);
                        self.sort_request = SortRequest::Throttled;
                    }
                }
            }
            _ => {}
        }

        if let Some(window) = self.window
            && egui_response.is_some_and(|response| response.repaint)
        {
            window.request_redraw();
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window {
            window.request_redraw();
        }
    }
}

impl<'window> ViewerApp<'window> {
    fn prepare_egui_frame(&mut self, window: &'window Window) -> Option<EguiFrame> {
        let raw_input = self.egui_state.as_mut()?.take_egui_input(window);
        let previous_options = self.render_options;
        let ctx = self.egui_ctx.clone();
        let mut actions = UiActions::default();
        let full_output = ctx.run(raw_input, |ctx| {
            actions = self.draw_control_panel(ctx);
        });

        if let Some(egui_state) = self.egui_state.as_mut() {
            egui_state.handle_platform_output(window, full_output.platform_output);
        }

        if actions.open_scene {
            self.open_scene_dialog(window);
        }
        if actions.apply_camera {
            self.apply_camera_preset(window);
        }
        if actions.screenshot {
            self.save_screenshot(window);
        }
        if actions.reset_camera {
            self.reset_camera(window);
        }
        if actions.request_sort || self.render_options.sh_degree != previous_options.sh_degree {
            self.sort_request = SortRequest::Immediate;
        }

        let pixels_per_point = full_output.pixels_per_point;
        let paint_jobs = self
            .egui_ctx
            .tessellate(full_output.shapes, pixels_per_point);
        let size = window.inner_size();
        Some(EguiFrame {
            textures_delta: full_output.textures_delta,
            paint_jobs,
            screen_descriptor: ScreenDescriptor {
                size_in_pixels: [size.width.max(1), size.height.max(1)],
                pixels_per_point,
            },
        })
    }

    fn draw_control_panel(&mut self, ctx: &egui::Context) -> UiActions {
        let mut actions = UiActions::default();
        egui::Window::new("SplatRS controls")
            .default_pos(egui::pos2(12.0, 12.0))
            .default_width(300.0)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label("Scene");
                ui.small(short_path(&self.args.model));
                ui.horizontal(|ui| {
                    if ui.button("Open scene...").clicked() {
                        actions.open_scene = true;
                    }
                    ui.label(format!("{} splats", self.scene.len()));
                });
                if let Some(message) = &self.status_message {
                    ui.small(message);
                }

                ui.horizontal(|ui| {
                    ui.label("Camera");
                    ui.add(egui::DragValue::new(&mut self.args.camera_index).range(0..=9999));
                    if ui.button("Apply").clicked() {
                        actions.apply_camera = true;
                    }
                });

                if ui.button("Screenshot").clicked() {
                    actions.screenshot = true;
                }

                ui.separator();
                ui.label(format!(
                    "{} / {} splats",
                    self.last_drawn_instances,
                    self.scene.len()
                ));
                ui.separator();
                if ui
                    .checkbox(&mut self.render_options.point_mode, "Point mode")
                    .changed()
                {
                    actions.request_sort = true;
                }

                ui.horizontal(|ui| {
                    ui.label("SH degree");
                    for degree in 0..=3 {
                        if ui
                            .selectable_value(
                                &mut self.render_options.sh_degree,
                                degree,
                                degree.to_string(),
                            )
                            .changed()
                        {
                            actions.request_sort = true;
                        }
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Tone");
                    ui.selectable_value(&mut self.render_options.tone_map, ToneMap::None, "None");
                    ui.selectable_value(
                        &mut self.render_options.tone_map,
                        ToneMap::Reinhard,
                        "Reinhard",
                    );
                    ui.selectable_value(&mut self.render_options.tone_map, ToneMap::Aces, "ACES");
                });

                ui.horizontal(|ui| {
                    ui.label("Background");
                    background_button(ui, &mut self.render_options.background, "Sky", SKY);
                    background_button(ui, &mut self.render_options.background, "Gray", GRAY);
                    background_button(ui, &mut self.render_options.background, "Dark", DARK);
                    background_button(ui, &mut self.render_options.background, "White", WHITE);
                });

                ui.separator();
                slider(
                    ui,
                    &mut self.render_options.opacity_scale,
                    0.05..=8.0,
                    "Opacity",
                );
                slider(
                    ui,
                    &mut self.render_options.splat_scale,
                    0.05..=4.0,
                    "Splat scale",
                );
                slider(
                    ui,
                    &mut self.render_options.max_splat_radius,
                    2.0..=240.0,
                    "Max radius",
                );
                slider(
                    ui,
                    &mut self.render_options.exposure,
                    0.05..=3.0,
                    "Exposure",
                );
                slider(
                    ui,
                    &mut self.render_options.saturation,
                    0.0..=2.0,
                    "Saturation",
                );
                slider(
                    ui,
                    &mut self.render_options.color_max,
                    0.1..=8.0,
                    "Color max",
                );
                slider(
                    ui,
                    &mut self.render_options.alpha_cutoff,
                    0.0..=0.05,
                    "Alpha cutoff",
                );
                slider(
                    ui,
                    &mut self.render_options.max_alpha,
                    0.1..=1.0,
                    "Max alpha",
                );
                slider(
                    ui,
                    &mut self.render_options.kernel_cutoff,
                    0.5..=16.0,
                    "Kernel cutoff",
                );
                slider(
                    ui,
                    &mut self.render_options.lowpass_pixels,
                    0.0..=4.0,
                    "Low-pass",
                );

                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Reset camera").clicked() {
                        actions.reset_camera = true;
                    }
                    if ui.button("Reset display").clicked() {
                        self.render_options = self.initial_render_options;
                        actions.request_sort = true;
                    }
                    if ui.button("Full resort").clicked() {
                        actions.request_sort = true;
                    }
                });
            });
        actions
    }

    fn save_screenshot(&mut self, window: &Window) {
        match self.write_screenshot(window) {
            Ok(path) => {
                self.status_message = Some(format!("Saved {}", path.display()));
                tracing::info!("saved screenshot to {}", path.display());
            }
            Err(error) => {
                let message = format!("Screenshot failed: {error:#}");
                tracing::error!("{message}");
                self.status_message = Some(message);
            }
        }
    }

    fn write_screenshot(&self, window: &Window) -> Result<PathBuf> {
        let camera = self
            .camera
            .as_ref()
            .context("camera is not initialized yet")?;
        let size = window.inner_size();
        let output_dir = std::env::current_dir()
            .context("failed to read current directory")?
            .join("screenshots");
        fs::create_dir_all(&output_dir)
            .with_context(|| format!("failed to create {}", output_dir.display()))?;
        let filename = format!(
            "splatrs-{}-{}.bmp",
            scene_slug(&self.args.model),
            timestamp_millis()
        );
        let output = output_dir.join(filename);
        headless::render_scene_to_bmp(
            &self.scene,
            camera,
            self.render_options,
            size.width.max(1),
            size.height.max(1),
            RenderBackend::GpuQuad,
            &output,
        )?;
        self.write_screenshot_metadata(&output, camera, size.width.max(1), size.height.max(1))?;
        Ok(output)
    }

    fn write_screenshot_metadata(
        &self,
        output: &Path,
        camera: &Camera,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let eye = camera.eye();
        let target = camera.target;
        let options = self.render_options;
        let mut text = String::new();
        writeln!(text, "image: {}", output.display())?;
        writeln!(text, "scene: {}", self.args.model.display())?;
        writeln!(text, "scene_label: {}", self.scene.source_label)?;
        writeln!(text, "splats: {}", self.scene.len())?;
        writeln!(text, "camera_index: {}", self.args.camera_index)?;
        writeln!(text, "size: {}x{}", width, height)?;
        writeln!(text, "eye: {:.6} {:.6} {:.6}", eye.x, eye.y, eye.z)?;
        writeln!(
            text,
            "target: {:.6} {:.6} {:.6}",
            target.x, target.y, target.z
        )?;
        writeln!(text, "yaw: {:.6}", camera.yaw)?;
        writeln!(text, "pitch: {:.6}", camera.pitch)?;
        writeln!(text, "distance: {:.6}", camera.distance)?;
        writeln!(text, "fovy_radians: {:.6}", camera.fovy_radians)?;
        writeln!(text, "point_mode: {}", options.point_mode)?;
        writeln!(text, "sh_degree: {}", options.sh_degree)?;
        writeln!(text, "opacity_scale: {:.6}", options.opacity_scale)?;
        writeln!(text, "splat_scale: {:.6}", options.splat_scale)?;
        writeln!(text, "max_splat_radius: {:.6}", options.max_splat_radius)?;
        writeln!(text, "exposure: {:.6}", options.exposure)?;
        writeln!(text, "saturation: {:.6}", options.saturation)?;
        writeln!(text, "color_max: {:.6}", options.color_max)?;
        writeln!(text, "alpha_cutoff: {:.6}", options.alpha_cutoff)?;
        writeln!(text, "max_alpha: {:.6}", options.max_alpha)?;
        writeln!(text, "kernel_cutoff: {:.6}", options.kernel_cutoff)?;
        writeln!(text, "lowpass_pixels: {:.6}", options.lowpass_pixels)?;
        writeln!(text, "tone_map: {:?}", options.tone_map)?;
        writeln!(text, "footprint: {:?}", options.footprint)?;
        writeln!(text, "radius_alpha: {:?}", options.radius_alpha)?;
        writeln!(
            text,
            "background: {:.6} {:.6} {:.6}",
            options.background[0], options.background[1], options.background[2]
        )?;
        fs::write(output.with_extension("txt"), text)
            .with_context(|| format!("failed to write metadata for {}", output.display()))?;
        Ok(())
    }

    fn open_scene_dialog(&mut self, window: &'window Window) {
        let mut dialog = rfd::FileDialog::new()
            .set_title("Open GraphDECO point_cloud.ply")
            .add_filter("PLY scene", &["ply"]);
        if let Some(parent) = self.args.model.parent() {
            dialog = dialog.set_directory(parent);
        }

        if let Some(path) = dialog.pick_file() {
            if let Err(error) = self.load_scene_from_path(path, window) {
                let message = format!("Open failed: {error:#}");
                tracing::error!("{message}");
                self.status_message = Some(message);
            }
        }
    }

    fn load_scene_from_path(&mut self, path: PathBuf, window: &'window Window) -> Result<()> {
        let scene = loader::load_scene(&path, self.args.filters.load_options(self.args.max_splats))
            .with_context(|| format!("failed to load {}", path.display()))?;
        let sh_degree = self.args.sh_degree.resolve(scene.detected_sh_degree());
        let mut next_options = self.render_options;
        next_options.sh_degree = sh_degree;

        let mut base_options = Self::render_options_from_args(&self.args, &scene);
        base_options.sh_degree = sh_degree;

        self.args.model = path;
        self.scene = scene;
        self.render_options = next_options;
        self.initial_render_options = base_options;
        self.rebuild_renderer_for_current_scene(window)?;
        self.status_message = Some(format!("Loaded {}", short_path(&self.args.model)));
        Ok(())
    }

    fn rebuild_renderer_for_current_scene(&mut self, window: &'window Window) -> Result<()> {
        let size = window.inner_size();
        let camera = self.make_initial_camera(size.width, size.height);
        let renderer = pollster::block_on(Renderer::new(
            window,
            &self.scene,
            &camera,
            Duration::from_millis(self.args.sort_interval_ms),
            self.args.interactive_max_splats,
        ))
        .context("failed to initialize renderer for the selected scene")?;
        let egui_renderer =
            EguiRenderer::new(renderer.device(), renderer.output_format(), None, 1, false);

        self.initial_camera = Some(camera);
        self.camera = Some(camera);
        self.renderer = Some(renderer);
        self.egui_renderer = Some(egui_renderer);
        self.dragging = false;
        self.last_cursor = None;
        self.last_drawn_instances = self.scene.len();
        self.sort_request = SortRequest::Immediate;
        Ok(())
    }

    fn apply_camera_preset(&mut self, window: &Window) {
        let size = window.inner_size();
        let next_camera = self.make_initial_camera(size.width, size.height);
        self.initial_camera = Some(next_camera);
        if let Some(camera) = self.camera.as_mut() {
            *camera = next_camera;
        }
        self.sort_request = SortRequest::Immediate;
        self.status_message = Some(format!("Applied camera {}", self.args.camera_index));
    }

    fn reset_camera(&mut self, window: &Window) {
        let size = window.inner_size();
        let next_camera = self
            .initial_camera
            .unwrap_or_else(|| self.make_initial_camera(size.width, size.height));
        if let Some(camera) = self.camera.as_mut() {
            *camera = next_camera;
        }
        self.sort_request = SortRequest::Immediate;
    }

    fn render_options_from_args(args: &ViewArgs, scene: &SplatScene) -> RenderOptions {
        RenderOptions {
            sh_degree: args.sh_degree.resolve(scene.detected_sh_degree()),
            opacity_scale: args.opacity_scale.clamp(0.05, 8.0),
            splat_scale: args.splat_scale.clamp(0.05, 12.0),
            max_splat_radius: args.max_splat_radius.clamp(2.0, 1024.0),
            kernel_cutoff: args.kernel_cutoff.clamp(0.5, 25.0),
            lowpass_pixels: args.lowpass_pixels.clamp(0.0, 16.0),
            alpha_cutoff: args.alpha_cutoff.clamp(0.0, 1.0),
            max_alpha: args.max_alpha.clamp(0.0, 1.0),
            color_max: args.color_max.clamp(0.001, 1024.0),
            saturation: args.saturation.clamp(0.0, 2.0),
            footprint: args.footprint.as_renderer(),
            radius_alpha: args.radius_alpha.as_renderer(),
            background: args.background.as_rgb(),
            exposure: args.exposure.clamp(0.05, 8.0),
            tone_map: args.tone_map.as_renderer(),
            lowpass_alpha_compensation: args.lowpass_alpha_compensation,
            ..RenderOptions::default()
        }
    }

    fn make_initial_camera(&self, width: u32, height: u32) -> Camera {
        let aspect = width.max(1) as f32 / height.max(1) as f32;
        match cameras::load_preset_for_model(&self.args.model, &self.scene, self.args.camera_index)
        {
            Ok(Some(preset)) => {
                tracing::info!(
                    "using camera preset {} from cameras.json",
                    self.args.camera_index
                );
                Camera::from_eye_target_up(
                    preset.eye,
                    preset.target,
                    preset.up,
                    self.scene.radius,
                    aspect,
                    preset.fovy_radians,
                )
            }
            Ok(None) => Camera::for_scene(self.scene.view_center, self.scene.view_radius, aspect),
            Err(error) => {
                tracing::warn!("could not load cameras.json preset: {error:#}");
                Camera::for_scene(self.scene.view_center, self.scene.view_radius, aspect)
            }
        }
    }
}

fn short_path(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| path.display().to_string())
}

fn scene_slug(path: &Path) -> String {
    for ancestor in path.ancestors() {
        let Some(name) = ancestor.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if name == "point_cloud" || name.starts_with("iteration_") {
            continue;
        }
        let stem = Path::new(name)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or(name);
        if !stem.is_empty() && stem != "point_cloud" {
            return sanitize_filename(stem);
        }
    }
    "scene".to_string()
}

fn sanitize_filename(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>();
    let sanitized = sanitized.trim_matches('-').to_string();
    if sanitized.is_empty() {
        "scene".to_string()
    } else {
        sanitized
    }
}

fn timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

const SKY: [f32; 3] = [0.72, 0.80, 0.92];
const GRAY: [f32; 3] = [0.18, 0.18, 0.18];
const DARK: [f32; 3] = [0.015, 0.017, 0.02];
const WHITE: [f32; 3] = [1.0, 1.0, 1.0];

fn slider(ui: &mut egui::Ui, value: &mut f32, range: std::ops::RangeInclusive<f32>, label: &str) {
    ui.add(egui::Slider::new(value, range).text(label));
}

fn background_button(ui: &mut egui::Ui, background: &mut [f32; 3], label: &str, value: [f32; 3]) {
    let selected = background
        .iter()
        .zip(value)
        .all(|(current, target)| (*current - target).abs() < 0.001);
    if ui.selectable_label(selected, label).clicked() {
        *background = value;
    }
}

#[derive(Debug)]
struct FrameCounter {
    last_report: Instant,
    frames: u32,
}

impl Default for FrameCounter {
    fn default() -> Self {
        Self {
            last_report: Instant::now(),
            frames: 0,
        }
    }
}

impl FrameCounter {
    fn tick(&mut self) -> Option<f32> {
        self.frames += 1;
        let elapsed = self.last_report.elapsed();
        if elapsed >= Duration::from_millis(500) {
            let fps = self.frames as f32 / elapsed.as_secs_f32();
            self.frames = 0;
            self.last_report = Instant::now();
            Some(fps)
        } else {
            None
        }
    }
}
