//! Backend-neutral progress events and cooperative cancellation.
//!
//! Model pipelines publish events through [`ProgressObserver`]. Renderers
//! (CLI, JSONL, GUI, or tests) decide how those events are displayed; the
//! inference core never depends on a particular terminal UI.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::thread;
use std::time::{Duration, Instant};

use serde::Serialize;

/// Observable high-level generation stages shared by Wan and LTX.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationStage {
    ValidateInputs,
    LoadComponents,
    EncodePrompt,
    PrepareLatents,
    Denoise,
    DecodeVae,
    PostProcess,
    EncodeVideo,
    Finalize,
}

impl GenerationStage {
    /// Stable machine-readable stage name for JSONL consumers.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ValidateInputs => "validate_inputs",
            Self::LoadComponents => "load_components",
            Self::EncodePrompt => "encode_prompt",
            Self::PrepareLatents => "prepare_latents",
            Self::Denoise => "denoise",
            Self::DecodeVae => "decode_vae",
            Self::PostProcess => "post_process",
            Self::EncodeVideo => "encode_video",
            Self::Finalize => "finalize",
        }
    }
}

/// Transformer passes performed for a single denoising step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DenoisePass {
    Conditional,
    Unconditional,
    Perturbed,
}

/// Structured event stream emitted by generation backends.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GenerationEvent {
    StageStarted {
        stage: GenerationStage,
        message: String,
    },
    StageProgress {
        stage: GenerationStage,
        current: u64,
        total: Option<u64>,
        message: Option<String>,
    },
    DenoiseStepStarted {
        step: usize,
        total_steps: usize,
        timestep: f64,
    },
    DenoisePassStarted {
        step: usize,
        pass: DenoisePass,
    },
    DenoiseStepFinished {
        step: usize,
        total_steps: usize,
        timestep: f64,
        elapsed_secs: f64,
    },
    Heartbeat {
        stage: GenerationStage,
        elapsed_secs: f64,
        message: String,
    },
    StageFinished {
        stage: GenerationStage,
        elapsed_secs: f64,
    },
    ConfigResolved {
        requested_width: usize,
        requested_height: usize,
        adjusted_width: usize,
        adjusted_height: usize,
        requested_frames: usize,
        adjusted_frames: usize,
        latent_shape: Vec<usize>,
        steps: usize,
        guidance_scale: f32,
        seed: Option<u64>,
        scheduler: String,
    },
    GenerationFinished {
        output: Option<String>,
        elapsed_secs: f64,
    },
    Warning {
        message: String,
    },
}

/// Receives structured generation events and can request cooperative cancel.
pub trait ProgressObserver: Send + Sync {
    /// Consume one event. Implementations should return quickly.
    fn on_event(&self, event: &GenerationEvent);

    /// Return `true` once the current operation should stop at a safe boundary.
    fn is_cancelled(&self) -> bool {
        false
    }

    /// Heartbeat interval for long synchronous operations. Zero disables it.
    fn heartbeat_interval(&self) -> Duration {
        Duration::ZERO
    }
}

/// Observer for library callers that do not need progress output.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopProgressObserver;

impl ProgressObserver for NoopProgressObserver {
    fn on_event(&self, _event: &GenerationEvent) {}
}

/// Thread-safe cooperative cancellation flag.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Create a clear cancellation token.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation. The current CUDA/CPU operation is allowed to
    /// finish; pipelines observe this flag at safe stage/pass boundaries.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Clear a token before reusing it for another generation.
    pub fn reset(&self) {
        self.cancelled.store(false, Ordering::Release);
    }
}

impl ProgressObserver for CancellationToken {
    fn on_event(&self, _event: &GenerationEvent) {}

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

/// Return the canonical Candle error used for cooperative cancellation.
pub fn cancellation_error() -> candle_core::Error {
    candle_core::Error::Msg("generation cancelled".to_string())
}

/// Stop at a safe boundary if an observer requested cancellation.
pub fn ensure_not_cancelled(observer: &dyn ProgressObserver) -> candle_core::Result<()> {
    if observer.is_cancelled() {
        Err(cancellation_error())
    } else {
        Ok(())
    }
}

/// Emit periodic heartbeat events while a synchronous model operation runs.
///
/// The operation itself is not interrupted mid-kernel. Cancellation is checked
/// by the caller immediately after this helper returns and between passes.
pub fn run_with_heartbeat<T, F>(
    observer: &Arc<dyn ProgressObserver>,
    stage: GenerationStage,
    message: impl Into<String>,
    operation: F,
) -> candle_core::Result<T>
where
    F: FnOnce() -> candle_core::Result<T>,
{
    let interval = observer.heartbeat_interval();
    if interval.is_zero() {
        return operation();
    }

    let (stop_sender, stop_receiver) = mpsc::channel();
    let observer_thread = Arc::clone(observer);
    let message = message.into();
    let started = Instant::now();
    let heartbeat = thread::spawn(move || {
        loop {
            match stop_receiver.recv_timeout(interval) {
                Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    observer_thread.on_event(&GenerationEvent::Heartbeat {
                        stage,
                        elapsed_secs: started.elapsed().as_secs_f64(),
                        message: message.clone(),
                    });
                }
            }
        }
    });

    let result = operation();
    let _ = stop_sender.send(());
    let _ = heartbeat.join();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_names_are_stable() {
        assert_eq!(GenerationStage::DecodeVae.as_str(), "decode_vae");
        assert_eq!(GenerationStage::EncodeVideo.as_str(), "encode_video");
    }

    #[test]
    fn cancellation_error_is_diagnostic() {
        assert_eq!(cancellation_error().to_string(), "generation cancelled");
    }
}
