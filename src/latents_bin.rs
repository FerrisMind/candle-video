//! Simple binary tensor IO with a u64 dims header.
//!
//! Format (little-endian):
//! - ndims: u64
//! - dims: ndims * u64
//! - data: f32 * product(dims)

use candle_core::{DType, Device, Result, Tensor};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

pub fn read_f32_tensor_with_header(
    path: impl AsRef<Path>,
    device: &Device,
) -> Result<(Vec<usize>, Tensor)> {
    let mut file = File::open(path)?;

    let mut u64_buf = [0u8; 8];
    file.read_exact(&mut u64_buf)?;
    let ndims = u64::from_le_bytes(u64_buf) as usize;

    let mut dims = Vec::with_capacity(ndims);
    for _ in 0..ndims {
        file.read_exact(&mut u64_buf)?;
        dims.push(u64::from_le_bytes(u64_buf) as usize);
    }

    let numel: usize = dims.iter().product();
    let mut data_bytes = vec![0u8; numel * 4];
    file.read_exact(&mut data_bytes)?;

    let data: Vec<f32> = data_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let tensor = Tensor::from_vec(data, dims.as_slice(), device)?;
    Ok((dims, tensor))
}

pub fn write_f32_tensor_with_header(path: impl AsRef<Path>, tensor: &Tensor) -> Result<()> {
    let mut file = File::create(path)?;

    let dims = tensor.dims().to_vec();
    let ndims = dims.len() as u64;
    file.write_all(&ndims.to_le_bytes())?;
    for d in &dims {
        file.write_all(&(*d as u64).to_le_bytes())?;
    }

    let flat = tensor.flatten_all()?.to_dtype(DType::F32)?;
    let data = flat.to_vec1::<f32>()?;
    for v in data {
        file.write_all(&v.to_le_bytes())?;
    }

    Ok(())
}
