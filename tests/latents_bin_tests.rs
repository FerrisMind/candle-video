use candle_core::{DType, Device, Result, Tensor};
use candle_video::{read_f32_tensor_with_header, write_f32_tensor_with_header};

#[test]
fn test_latents_bin_roundtrip_f32() -> Result<()> {
    let device = Device::Cpu;
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let path = tmp.path();

    let input = Tensor::randn(0f32, 1.0, (1, 3, 2, 4, 5), &device)?.to_dtype(DType::F32)?;
    write_f32_tensor_with_header(path, &input)?;

    let (dims, loaded) = read_f32_tensor_with_header(path, &device)?;
    assert_eq!(dims, vec![1, 3, 2, 4, 5]);
    assert_eq!(loaded.dims(), input.dims());

    let diff = input.sub(&loaded)?.abs()?.max_all()?.to_scalar::<f32>()?;
    assert!(diff < 1e-6, "max diff {diff}");
    Ok(())
}
