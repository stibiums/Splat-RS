use std::time::{Duration, Instant};

use anyhow::Result;
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
    cli::ViewArgs,
    loader,
    renderer::{RenderOptions, Renderer, ToneMap},
    scene::SplatScene,
};

pub fn run(args: ViewArgs) -> Result<()> {
    let scene = loader::load_scene(&args.model, args.filters.load_options(args.max_splats))?;
    tracing::info!(
        "loaded {} splats from {} (bounds {:?} .. {:?}, file SH degree {}, requested SH degree {})",
        scene.len(),
        scene.source_label,
        scene.bounds_min,
        scene.bounds_max,
        scene.detected_sh_degree(),
        args.sh_degree.as_u32()
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
    dragging: bool,
    last_cursor: Option<PhysicalPosition<f64>>,
    force_sort: bool,
    frame_counter: FrameCounter,
}

impl<'window> ViewerApp<'window> {
    fn new(args: ViewArgs, scene: SplatScene) -> Self {
        let render_options = RenderOptions {
            sh_degree: args.sh_degree.as_u32(),
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
        };

        Self {
            args,
            scene,
            window: None,
            window_id: None,
            renderer: None,
            camera: None,
            initial_camera: None,
            render_options,
            dragging: false,
            last_cursor: None,
            force_sort: true,
            frame_counter: FrameCounter::default(),
        }
    }

    fn render(&mut self) {
        let (Some(window), Some(renderer), Some(camera)) =
            (self.window, self.renderer.as_mut(), self.camera.as_ref())
        else {
            return;
        };

        if let Err(error) =
            renderer.render(&self.scene, camera, self.render_options, self.force_sort)
        {
            tracing::error!("{error:#}");
            window.set_title(&format!("SplatRS - render error: {error}"));
        }
        self.force_sort = false;

        if let Some(fps) = self.frame_counter.tick() {
            window.set_title(&format!(
                "SplatRS - {} splats - {:.1} FPS - {} - opacity {:.2} - scale {:.2} - radius {:.0}px - exposure {:.2} - {:?} - SH d{} - {}",
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
        let renderer = pollster::block_on(Renderer::new(window, &self.scene, &camera))
            .expect("failed to initialize renderer");

        self.window = Some(window);
        self.window_id = Some(window_id);
        self.initial_camera = Some(camera);
        self.camera = Some(camera);
        self.renderer = Some(renderer);
        self.force_sort = true;
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
                self.force_sort = true;
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                    PhysicalKey::Code(KeyCode::KeyP) => {
                        self.render_options.point_mode = !self.render_options.point_mode;
                    }
                    PhysicalKey::Code(KeyCode::KeyO) => {
                        self.render_options.opacity_scale =
                            (self.render_options.opacity_scale * 1.15).min(8.0);
                    }
                    PhysicalKey::Code(KeyCode::KeyI) => {
                        self.render_options.opacity_scale =
                            (self.render_options.opacity_scale / 1.15).max(0.05);
                    }
                    PhysicalKey::Code(KeyCode::Equal)
                    | PhysicalKey::Code(KeyCode::NumpadAdd)
                    | PhysicalKey::Code(KeyCode::BracketRight) => {
                        self.render_options.splat_scale =
                            (self.render_options.splat_scale * 1.15).min(12.0);
                    }
                    PhysicalKey::Code(KeyCode::Minus)
                    | PhysicalKey::Code(KeyCode::NumpadSubtract)
                    | PhysicalKey::Code(KeyCode::BracketLeft) => {
                        self.render_options.splat_scale =
                            (self.render_options.splat_scale / 1.15).max(0.05);
                    }
                    PhysicalKey::Code(KeyCode::Period) => {
                        self.render_options.max_splat_radius =
                            (self.render_options.max_splat_radius * 1.15).min(1024.0);
                    }
                    PhysicalKey::Code(KeyCode::Comma) => {
                        self.render_options.max_splat_radius =
                            (self.render_options.max_splat_radius / 1.15).max(2.0);
                    }
                    PhysicalKey::Code(KeyCode::KeyE) => {
                        self.render_options.exposure =
                            (self.render_options.exposure * 1.15).min(8.0);
                    }
                    PhysicalKey::Code(KeyCode::KeyD) => {
                        self.render_options.exposure =
                            (self.render_options.exposure / 1.15).max(0.05);
                    }
                    PhysicalKey::Code(KeyCode::KeyT) => {
                        self.render_options.tone_map = match self.render_options.tone_map {
                            ToneMap::None => ToneMap::Reinhard,
                            ToneMap::Reinhard => ToneMap::Aces,
                            ToneMap::Aces => ToneMap::None,
                        };
                    }
                    PhysicalKey::Code(KeyCode::KeyR) => {
                        if let Some(window) = self.window {
                            let size = window.inner_size();
                            let next_camera = self.initial_camera.unwrap_or_else(|| {
                                self.make_initial_camera(size.width, size.height)
                            });
                            if let Some(camera) = self.camera.as_mut() {
                                *camera = next_camera;
                            }
                            self.force_sort = true;
                        }
                    }
                    PhysicalKey::Code(KeyCode::Digit0) | PhysicalKey::Code(KeyCode::Numpad0) => {
                        self.render_options.sh_degree = 0;
                        self.force_sort = true;
                    }
                    PhysicalKey::Code(KeyCode::Digit1) | PhysicalKey::Code(KeyCode::Numpad1) => {
                        self.render_options.sh_degree = 1;
                        self.force_sort = true;
                    }
                    PhysicalKey::Code(KeyCode::Digit2) | PhysicalKey::Code(KeyCode::Numpad2) => {
                        self.render_options.sh_degree = 2;
                        self.force_sort = true;
                    }
                    PhysicalKey::Code(KeyCode::Digit3) | PhysicalKey::Code(KeyCode::Numpad3) => {
                        self.render_options.sh_degree = 3;
                        self.force_sort = true;
                    }
                    _ => {}
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.dragging = state == ElementState::Pressed;
                if !self.dragging {
                    self.last_cursor = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.dragging {
                    if let (Some(previous), Some(camera)) = (self.last_cursor, self.camera.as_mut())
                    {
                        camera.orbit(
                            (position.x - previous.x) as f32,
                            (position.y - previous.y) as f32,
                        );
                        self.force_sort = true;
                    }
                    self.last_cursor = Some(position);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(position) => (position.y as f32) / 40.0,
                };
                if let Some(camera) = self.camera.as_mut() {
                    camera.zoom(scroll);
                    self.force_sort = true;
                }
            }
            _ => {}
        }

        if let Some(window) = self.window {
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
