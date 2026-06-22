//! Load exported PyTorch initial latents (`scripts/wan/export_latents.py`).

use std::path::Path;

use candle_core::{Device, Result, Tensor};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct LatentsFixture {
    shape: Vec<usize>,
    data: Vec<f32>,
}

pub fn load_latents_json(path: impl AsRef<Path>, device: &Device) -> Result<Tensor> {
    let raw = std::fs::read_to_string(path.as_ref()).map_err(candle_core::Error::wrap)?;
    let fixture: LatentsFixture = serde_json::from_str(&raw).map_err(candle_core::Error::wrap)?;
    let v = fixture.data;
    Tensor::from_vec(v, fixture.shape.as_slice(), device)
}
