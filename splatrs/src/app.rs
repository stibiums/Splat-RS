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
    cli::ViewArgs,
    loader,
    renderer::{RenderOptions, Renderer},
    scene::SplatScene,
};

pub fn run(args: ViewArgs) -> Result<()> {
    let scene = loader::load_scene(&args.model, args.max_splats)?;
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
    render_options: RenderOptions,
    dragging: bool,
    last_cursor: Option<PhysicalPosition<f64>>,
    force_sort: bool,
    frame_counter: FrameCounter,
}

impl<'window> ViewerApp<'window> {
    fn new(args: ViewArgs, scene: SplatScene) -> Self {
        Self {
            args,
            scene,
            window: None,
            window_id: None,
            renderer: None,
            camera: None,
            render_options: RenderOptions::default(),
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
                "SplatRS - {} splats - {:.1} FPS - {} - opacity {:.2} - scale {:.2} - SH d{} - {}",
                self.scene.len(),
                fps,
                if self.render_options.point_mode {
                    "points"
                } else {
                    "splats"
                },
                self.render_options.opacity_scale,
                self.render_options.splat_scale,
                self.args.sh_degree.as_u32(),
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
        let camera = Camera::for_scene(
            self.scene.center,
            self.scene.radius,
            size.width.max(1) as f32 / size.height.max(1) as f32,
        );
        let renderer = pollster::block_on(Renderer::new(window, &self.scene, &camera))
            .expect("failed to initialize renderer");

        self.window = Some(window);
        self.window_id = Some(window_id);
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
                    PhysicalKey::Code(KeyCode::KeyR) => {
                        if let (Some(window), Some(camera)) = (self.window, self.camera.as_mut()) {
                            let size = window.inner_size();
                            *camera = Camera::for_scene(
                                self.scene.center,
                                self.scene.radius,
                                size.width.max(1) as f32 / size.height.max(1) as f32,
                            );
                            self.force_sort = true;
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::MouseInput { state, button, .. } if button == MouseButton::Left => {
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
