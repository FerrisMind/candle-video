use candle_video::{RunManifest, write_run_manifest};

#[test]
fn run_manifest_is_written_as_atomic_json() {
    let temp = tempfile::tempdir().expect("tempdir");
    let path = temp.path().join("run.json");
    let manifest = RunManifest {
        model: "Wan2.1-T2V-1.3B".to_string(),
        backend: "cuda".to_string(),
        device: "CUDA:0".to_string(),
        dtype: "f16".to_string(),
        width: 32,
        height: 32,
        frames: 1,
        fps: 16,
        steps: 1,
        guidance_scale: 1.0,
        seed: Some(42),
        scheduler: "flow-match-euler".to_string(),
        total_seconds: 1.5,
        peak_vram_bytes: None,
        output: "run.mp4".to_string(),
    };

    write_run_manifest(&path, &manifest).expect("write run manifest");

    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).expect("read"))
        .expect("valid run manifest JSON");
    assert_eq!(value["model"], "Wan2.1-T2V-1.3B");
    assert_eq!(value["seed"], 42);
    assert!(!std::path::PathBuf::from(format!("{}.partial", path.display())).exists());
}
