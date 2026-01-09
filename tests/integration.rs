//! Integration tests for candle-video
//!
//! NOTE: This test uses legacy imports and is disabled.
//! Use verify_*_parity.rs tests for current parity testing.

#![cfg(feature = "legacy_tests")]

use candle_core::Device;
use candle_video::*;

#[test]
fn test_config_validation() {
    // Valid config
    let config = InferenceConfig::new(17, 512, 768, 42);
    assert!(config.is_ok());

    // Invalid frame count
    let config = InferenceConfig::new(16, 512, 768, 42);
    assert!(config.is_err());

    // Invalid dimensions
    let config = InferenceConfig::new(17, 500, 768, 42);
    assert!(config.is_err());
}

#[test]
fn test_weight_loader() {
    let loader = WeightLoader::new(Device::Cpu, candle_core::DType::F32);
    assert_eq!(loader.dtype(), candle_core::DType::F32);
}

#[test]
fn test_scheduler() {
    use candle_video::SchedulerConfig;
    let config = SchedulerConfig::default();
    let scheduler = RectifiedFlowScheduler::new(config);

    assert_eq!(scheduler.num_steps(), 50);
    assert_eq!(scheduler.guidance_scale(), 3.0);
}
