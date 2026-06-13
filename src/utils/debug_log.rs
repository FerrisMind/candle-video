//! NDJSON debug logging for agent debug sessions.

use std::fs::OpenOptions;
use std::io::Write;

use candle_core::{Result, Tensor};

const LOG_PATH: &str = "/home/mod479711/Downloads/.cursor/debug-f0388a.log";

pub fn debug_log(hypothesis_id: &str, location: &str, message: &str, data: serde_json::Value) {
    let line = serde_json::json!({
        "sessionId": "f0388a",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0),
    });
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(LOG_PATH) {
        let _ = writeln!(f, "{}", line);
    }
}

pub fn tensor_stats(name: &str, t: &Tensor) -> Result<serde_json::Value> {
    let flat = t.flatten_all()?.to_dtype(candle_core::DType::F32)?;
    let v = flat.to_vec1::<f32>()?;
    let (min, max) = v
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &x| {
            (mn.min(x), mx.max(x))
        });
    let mean = v.iter().sum::<f32>() / v.len().max(1) as f32;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len().max(1) as f32;
    Ok(serde_json::json!({
        "name": name,
        "shape": t.dims(),
        "min": min,
        "max": max,
        "mean": mean,
        "std": var.sqrt(),
        "first8": &v[..v.len().min(8)],
    }))
}
