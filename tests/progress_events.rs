use std::sync::{Arc, Mutex};
use std::time::Duration;

use candle_video::{
    CancellationToken, DenoisePass, GenerationEvent, GenerationStage, NoopProgressObserver,
    ProgressObserver, run_with_heartbeat,
};

#[test]
fn generation_events_serialize_with_stable_type_names() {
    let event = GenerationEvent::DenoiseStepStarted {
        step: 2,
        total_steps: 30,
        timestep: 766.7,
    };
    let value = serde_json::to_value(event).expect("serialize generation event");
    assert_eq!(value["type"], "denoise_step_started");
    assert_eq!(value["step"], 2);
    assert_eq!(value["total_steps"], 30);
}

#[test]
fn observer_can_report_cancellation_without_renderer_state() {
    let token = CancellationToken::new();
    let observer: Arc<dyn ProgressObserver> = Arc::new(token.clone());
    assert!(!observer.is_cancelled());

    token.cancel();

    assert!(observer.is_cancelled());
}

#[test]
fn noop_observer_is_safe_for_library_callers() {
    let observer = NoopProgressObserver;
    observer.on_event(&GenerationEvent::DenoisePassStarted {
        step: 1,
        pass: DenoisePass::Conditional,
    });
    assert!(!observer.is_cancelled());
    assert_eq!(GenerationStage::Denoise.as_str(), "denoise");
}

struct RecordingObserver {
    events: Mutex<Vec<GenerationEvent>>,
}

impl ProgressObserver for RecordingObserver {
    fn on_event(&self, event: &GenerationEvent) {
        self.events.lock().expect("events lock").push(event.clone());
    }

    fn heartbeat_interval(&self) -> Duration {
        Duration::from_millis(5)
    }
}

#[test]
fn heartbeat_is_emitted_during_a_long_operation() {
    let observer = Arc::new(RecordingObserver {
        events: Mutex::new(Vec::new()),
    });
    let observer_dyn: Arc<dyn ProgressObserver> = observer.clone();
    run_with_heartbeat(
        &observer_dyn,
        GenerationStage::Denoise,
        "test operation",
        || {
            std::thread::sleep(Duration::from_millis(20));
            Ok::<_, candle_core::Error>(())
        },
    )
    .expect("heartbeat operation");

    assert!(
        observer
            .events
            .lock()
            .expect("events lock")
            .iter()
            .any(|event| matches!(event, GenerationEvent::Heartbeat { .. }))
    );
}
