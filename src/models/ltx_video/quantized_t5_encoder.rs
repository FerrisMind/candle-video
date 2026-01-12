//! Quantized T5 Encoder Model for GGUF format
//!
//! Re-exports from common interfaces with LTX-specific presets.

// Re-export everything from common interface
pub use crate::interfaces::quantized_t5_encoder::{QuantizedT5Config, QuantizedT5EncoderModel};

// Re-export T5EncoderConfig for convenience
pub use crate::interfaces::t5_encoder::T5EncoderConfig;

// Legacy alias for backward compatibility
pub type T5EncoderConfigLegacy = QuantizedT5Config;
