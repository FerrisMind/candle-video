//! Shared terminal/JSONL progress renderer for the LTX and Wan examples.

use std::fs::{File, OpenOptions};
use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::time::Duration;

use candle_video::{
    CancellationToken, DenoisePass, GenerationEvent, GenerationStage, ProgressObserver,
};
use clap::ValueEnum;
use indicatif::{ProgressBar, ProgressStyle};

/// Terminal progress policy.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum ProgressMode {
    #[default]
    Auto,
    Always,
    Never,
}

/// Human or machine-readable event output.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum LogFormat {
    #[default]
    Human,
    Json,
}

/// Renderer configuration shared by both model examples.
#[derive(Debug, Clone, Default)]
pub struct CliProgressOptions {
    pub progress: ProgressMode,
    pub log_format: LogFormat,
    pub quiet: bool,
    pub verbose: u8,
    pub timings: bool,
    pub diagnostics: bool,
    pub gpu_stats: bool,
    pub heartbeat_seconds: u64,
    pub events_jsonl: Option<PathBuf>,
}

struct RenderState {
    denoise_bar: Option<ProgressBar>,
    last_stage: Option<GenerationStage>,
}

/// Progress observer used by the example CLIs.
pub struct CliProgressObserver {
    options: CliProgressOptions,
    cancellation: CancellationToken,
    state: Mutex<RenderState>,
    event_file: Mutex<Option<File>>,
    peak_vram_bytes: AtomicU64,
}

impl CliProgressObserver {
    /// Create a renderer and truncate an optional JSONL event file.
    pub fn new(options: CliProgressOptions) -> io::Result<Arc<Self>> {
        let event_file = options
            .events_jsonl
            .as_ref()
            .map(|path| {
                if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
                    std::fs::create_dir_all(parent)?;
                }
                OpenOptions::new()
                    .create(true)
                    .truncate(true)
                    .write(true)
                    .open(path)
            })
            .transpose()?;
        Ok(Arc::new(Self {
            options,
            cancellation: CancellationToken::new(),
            state: Mutex::new(RenderState {
                denoise_bar: None,
                last_stage: None,
            }),
            event_file: Mutex::new(event_file),
            peak_vram_bytes: AtomicU64::new(0),
        }))
    }

    /// Return the largest best-effort VRAM usage observed so far.
    pub fn peak_vram_bytes(&self) -> Option<u64> {
        match self.peak_vram_bytes.load(Ordering::Relaxed) {
            0 => None,
            value => Some(value),
        }
    }

    fn record_gpu_snapshot(&self) -> Option<String> {
        let snapshot = candle_video::profiling::vram::snapshot_nvidia_smi()?;
        let used_mib = snapshot
            .split(',')
            .nth(2)
            .and_then(|value| value.trim().parse::<u64>().ok())?;
        let used_bytes = used_mib.saturating_mul(1024 * 1024);
        self.peak_vram_bytes
            .fetch_max(used_bytes, Ordering::Relaxed);
        Some(snapshot)
    }

    /// Install one process-wide Ctrl+C handler that requests a safe cancel.
    pub fn install_ctrlc(observer: &Arc<Self>) -> Result<(), ctrlc::Error> {
        let observer = Arc::clone(observer);
        ctrlc::set_handler(move || {
            observer.cancel();
            eprintln!("Cancellation requested; finishing the current operation safely …");
        })
    }

    /// Request cancellation without terminating an in-flight CUDA kernel.
    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    fn human_enabled(&self) -> bool {
        if self.options.quiet || matches!(self.options.progress, ProgressMode::Never) {
            return false;
        }
        match self.options.progress {
            ProgressMode::Always => true,
            ProgressMode::Auto | ProgressMode::Never => io::stderr().is_terminal(),
        }
    }

    fn write_event_file(&self, line: &str) {
        if let Ok(mut file) = self.event_file.lock()
            && let Some(file) = file.as_mut()
        {
            let _ = writeln!(file, "{line}");
            let _ = file.flush();
        }
    }

    fn emit_json(&self, event: &GenerationEvent) {
        let Ok(line) = serde_json::to_string(event) else {
            return;
        };
        eprintln!("{line}");
        self.write_event_file(&line);
    }

    fn print_line(&self, state: &RenderState, message: impl AsRef<str>) {
        if let Some(bar) = &state.denoise_bar {
            bar.println(message.as_ref());
        } else {
            eprintln!("{}", message.as_ref());
        }
    }

    fn ensure_denoise_bar(&self, state: &mut RenderState, total: usize) -> ProgressBar {
        if let Some(bar) = &state.denoise_bar {
            return bar.clone();
        }
        let bar = ProgressBar::new(total as u64);
        if let Ok(style) = ProgressStyle::with_template(
            "      {prefix} [{bar:32.cyan/blue}] {pos}/{len} {percent:>3}% · {msg} · ETA {eta}",
        ) {
            bar.set_style(style.progress_chars("█░"));
        }
        bar.set_prefix("Denoising");
        state.denoise_bar = Some(bar.clone());
        bar
    }

    fn render_human(&self, event: &GenerationEvent) {
        if !self.human_enabled() {
            return;
        }
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        match event {
            GenerationEvent::StageStarted { stage, message } => {
                state.last_stage = Some(*stage);
                self.print_line(&state, format!("[{}] {message}", stage.as_str()));
            }
            GenerationEvent::StageProgress {
                stage,
                current,
                total,
                message,
            } if *stage != GenerationStage::Denoise => {
                let suffix = message.as_deref().unwrap_or("");
                let progress = total.map_or_else(
                    || format!("{current}"),
                    |total| format!("{current}/{total}"),
                );
                self.print_line(&state, format!("      {progress} {suffix}"));
            }
            GenerationEvent::DenoiseStepStarted {
                step,
                total_steps,
                timestep,
            } => {
                let bar = self.ensure_denoise_bar(&mut state, *total_steps);
                bar.set_position((*step).saturating_sub(1) as u64);
                bar.set_message(format!("timestep {timestep:.1}"));
            }
            GenerationEvent::DenoisePassStarted { step, pass } => {
                if self.options.verbose > 0 || self.options.diagnostics {
                    let label = match pass {
                        DenoisePass::Conditional => "conditional",
                        DenoisePass::Unconditional => "unconditional",
                        DenoisePass::Perturbed => "perturbed",
                    };
                    self.print_line(
                        &state,
                        format!("      step {step} · {label} transformer pass"),
                    );
                }
            }
            GenerationEvent::DenoiseStepFinished {
                step, elapsed_secs, ..
            } => {
                let bar = self.ensure_denoise_bar(&mut state, *step);
                bar.set_position(*step as u64);
                if self.options.timings {
                    bar.set_message(format!("last step {elapsed_secs:.1}s"));
                }
            }
            GenerationEvent::Heartbeat {
                stage,
                elapsed_secs,
                message,
            } => {
                let mut line = format!(
                    "      {} · elapsed {}",
                    message,
                    format_elapsed(*elapsed_secs)
                );
                if self.options.gpu_stats
                    && let Some(stats) = self.record_gpu_snapshot()
                {
                    line.push_str(&format!(" · GPU {stats}"));
                }
                if self.options.diagnostics || *stage != GenerationStage::Denoise {
                    self.print_line(&state, line);
                } else if let Some(bar) = &state.denoise_bar {
                    bar.println(line);
                }
            }
            GenerationEvent::StageFinished {
                stage,
                elapsed_secs,
            } => {
                if let Some(bar) = state.denoise_bar.take()
                    && *stage == GenerationStage::Denoise
                {
                    bar.finish_and_clear();
                }
                if self.options.timings {
                    self.print_line(
                        &state,
                        format!("[{}] done in {:.2}s", stage.as_str(), elapsed_secs),
                    );
                } else {
                    self.print_line(&state, format!("[{}] done", stage.as_str()));
                }
            }
            GenerationEvent::ConfigResolved {
                requested_width,
                requested_height,
                adjusted_width,
                adjusted_height,
                requested_frames,
                adjusted_frames,
                latent_shape,
                steps,
                guidance_scale,
                seed,
                scheduler,
            } => {
                self.print_line(
                    &state,
                    format!(
                        "      config: {}x{} → {}x{}, frames {} → {}, latent {:?}, {} steps, CFG {:.2}, seed {:?}, scheduler {}",
                        requested_width,
                        requested_height,
                        adjusted_width,
                        adjusted_height,
                        requested_frames,
                        adjusted_frames,
                        latent_shape,
                        steps,
                        guidance_scale,
                        seed,
                        scheduler
                    ),
                );
            }
            GenerationEvent::GenerationFinished {
                output,
                elapsed_secs,
            } => {
                let output = output.as_deref().unwrap_or("(not written)");
                self.print_line(
                    &state,
                    format!(
                        "Generation completed: {output} · total {}",
                        format_elapsed(*elapsed_secs)
                    ),
                );
            }
            GenerationEvent::Warning { message } => {
                self.print_line(&state, format!("warning: {message}"))
            }
            _ => {}
        }
    }
}

impl ProgressObserver for CliProgressObserver {
    fn on_event(&self, event: &GenerationEvent) {
        if self.options.gpu_stats
            && matches!(event, GenerationEvent::Heartbeat { .. })
            && (!matches!(self.options.log_format, LogFormat::Human) || !self.human_enabled())
        {
            let _ = self.record_gpu_snapshot();
        }
        if matches!(self.options.log_format, LogFormat::Json) {
            self.emit_json(event);
        } else {
            if let Ok(line) = serde_json::to_string(event) {
                self.write_event_file(&line);
            }
            self.render_human(event);
        }
    }

    fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }

    fn heartbeat_interval(&self) -> Duration {
        Duration::from_secs(self.options.heartbeat_seconds)
    }
}

fn format_elapsed(seconds: f64) -> String {
    if seconds < 60.0 {
        return format!("{seconds:.1}s");
    }
    let total = seconds.round() as u64;
    format!("{:02}:{:02}", total / 60, total % 60)
}
