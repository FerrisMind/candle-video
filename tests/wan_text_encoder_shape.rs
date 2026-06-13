//! UMT5 config tests (full weight load deferred to optional parity examples).

use candle_video::models::wan::Umt5EncoderConfig;

#[test]
fn umt5_config_wan21_matches_checkpoint() {
    let cfg = Umt5EncoderConfig::wan21_t2v_13b();
    assert_eq!(cfg.d_model, 4096);
    assert_eq!(cfg.num_layers, 24);
    assert_eq!(cfg.vocab_size, 256_384);

    let t5 = cfg.to_candle_t5_config();
    assert!(t5.feed_forward_proj.gated);
    assert_eq!(t5.num_layers, 24);
    assert!(t5.is_encoder_decoder);
}
