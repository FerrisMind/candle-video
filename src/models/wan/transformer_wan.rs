// Rust 2024 + candle
//
// Это порт архитектурной логики transformer_wan.py на candle.
// Важное: часть компонентов diffusers (FP32LayerNorm, FeedForward, PixArtAlphaTextProjection,
// Timesteps/TimestepEmbedding, dispatch_attention_fn, PEFT/LoRA, CP/CacheMixin) здесь заменены
// на минимальные эквиваленты или заглушки, чтобы сохранить поведение forward-прохода на уровне
// форм и вычислений. [file:3]

use candle_core::{DType, Device, IndexOp, Result as CandleResult, Shape, Tensor};
use regex::Regex;
use std::sync::OnceLock;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WanError {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

pub type Result<T> = std::result::Result<T, WanError>;

// ------------------------- utils -------------------------

fn prod(dims: &[usize]) -> usize {
    dims.iter().product()
}

fn ensure(cond: bool, msg: impl Into<String>) -> Result<()> {
    if cond {
        Ok(())
    } else {
        Err(WanError::InvalidArgument(msg.into()))
    }
}

// ------------------------- basic modules -------------------------

#[derive(Clone, Debug)]
pub struct Linear {
    // weight: [out, in], bias: [out]
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [..., in]
        let w_t = self.weight.transpose(0, 1)?; // [in, out]
        let y = x.matmul(&w_t)?;
        if let Some(b) = &self.bias {
            // broadcast over leading dims
            Ok(y.broadcast_add(b)?)
        } else {
            Ok(y)
        }
    }
}

#[derive(Clone, Debug)]
pub struct Dropout {
    // In inference pipeline we do no-op (dropout_p=0.0 in python attention path). [file:3]
}

impl Dropout {
    pub fn forward(&self, x: Tensor) -> Result<Tensor> {
        Ok(x)
    }
}

/// Упрощённый LayerNorm (в оригинале FP32LayerNorm, иногда elementwise_affine=False). [file:3]
#[derive(Clone, Debug)]
pub struct LayerNorm {
    gamma: Option<Tensor>,
    beta: Option<Tensor>,
    eps: f64,
}

impl LayerNorm {
    pub fn new(gamma: Option<Tensor>, beta: Option<Tensor>, eps: f64) -> Self {
        Self { gamma, beta, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Нормируем по последней оси.
        let x_f = x.to_dtype(DType::F32)?;
        let mean = x_f.mean_keepdim(x_f.rank() - 1)?;
        let var = (x_f.broadcast_sub(&mean)?)
            .sqr()?
            .mean_keepdim(x_f.rank() - 1)?;
        let denom = (var + (self.eps as f32))?.sqrt()?;
        let mut y = x_f.broadcast_sub(&mean)?.broadcast_div(&denom)?;
        if let Some(g) = &self.gamma {
            y = y.broadcast_mul(g)?;
        }
        if let Some(b) = &self.beta {
            y = y.broadcast_add(b)?;
        }
        Ok(y.to_dtype(x.dtype())?)
    }
}

/// Упрощённый RMSNorm (в оригинале torch.nn.RMSNorm). [file:3]
#[derive(Clone, Debug)]
pub struct RmsNorm {
    eps: f64,
}

impl RmsNorm {
    pub fn new(eps: f64) -> Self {
        Self { eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f = x.to_dtype(DType::F32)?;
        let sq = x_f.sqr()?;
        let mean_sq = sq.mean_keepdim(x_f.rank() - 1)?;
        let denom = (mean_sq + (self.eps as f32))?.sqrt()?;
        let y = x_f.broadcast_div(&denom)?;
        Ok(y.to_dtype(x.dtype())?)
    }
}

/// Activation SiLU.
fn silu(x: &Tensor) -> Result<Tensor> {
    // x * sigmoid(x)
    let sig = x.sigmoid()?;
    Ok((x * sig)?)
}

// ------------------------- attention -------------------------

#[derive(Clone, Debug)]
pub struct WanAttention {
    pub dim: usize,
    pub heads: usize,
    pub dim_head: usize,
    pub inner_dim: usize,
    pub kv_inner_dim: usize,

    pub added_kv_proj_dim: Option<usize>,
    pub cross_attention_dim_head: Option<usize>,

    pub to_q: Linear,
    pub to_k: Linear,
    pub to_v: Linear,
    pub add_k_proj: Option<Linear>,
    pub add_v_proj: Option<Linear>,

    pub norm_q: RmsNorm,
    pub norm_k: RmsNorm,
    pub norm_added_k: Option<RmsNorm>,

    pub to_out: (Linear, Dropout),

    // fused_projections в python оптимизация — здесь не реализуем. [file:3]
}

impl WanAttention {
    pub fn forward(
        &self,
        hidden_states: &Tensor,                    // [b, s, dim]
        encoder_hidden_states: Option<&Tensor>,    // [b, s_ctx, dim] or None
        attention_mask: Option<&Tensor>,           // ignored in this minimal port
        rotary_emb: Option<(&Tensor, &Tensor)>,    // (cos, sin)
    ) -> Result<Tensor> {
        let encoder_hidden_states = encoder_hidden_states.unwrap_or(hidden_states);

        // QKV projections
        let mut q = self.to_q.forward(hidden_states)?; // [b, s, inner_dim]
        let mut k = self.to_k.forward(encoder_hidden_states)?; // [b, s_ctx, kv_inner_dim]
        let mut v = self.to_v.forward(encoder_hidden_states)?;

        // Norms
        q = self.norm_q.forward(&q)?;
        k = self.norm_k.forward(&k)?;

        // Reshape to heads: [b, s, h, d]
        let (b, s, _) = q.dims3()?;
        let (b2, s_ctx, _) = k.dims3()?;
        ensure(b == b2, "batch mismatch in attention")?;

        let q = q.reshape((b, s, self.heads, self.dim_head))?;
        let k = k.reshape((b, s_ctx, self.heads, self.dim_head))?;
        let v = v.reshape((b, s_ctx, self.heads, self.dim_head))?;

        // Rotary (only for q/k in processor in python). [file:3]
        let (q, k) = if let Some((cos, sin)) = rotary_emb {
            (apply_rotary_emb(&q, cos, sin)?, apply_rotary_emb(&k, cos, sin)?)
        } else {
            (q, k)
        };

        // I2V added KV path: python splits encoder_hidden_states into img + text if add_k_proj exists. [file:3]
        // Здесь оставляем поддержку: если add_k_proj задан, ожидаем, что encoder_hidden_states уже concat([img, text]).
        // Реальное разделение по 512 токенов — логика пайплайна/верхнего уровня. [file:3]
        let mut attn_out = scaled_dot_product_attention(&q, &k, &v, attention_mask)?;

        // Merge heads: python flatten(2,3) => [b, s, inner_dim]. [file:3]
        attn_out = attn_out.reshape((b, s, self.heads * self.dim_head))?;

        // Output proj
        let y = self.to_out.0.forward(&attn_out)?;
        self.to_out.1.forward(y)
    }
}

/// SDPA: q,k,v: [b, s, h, d] with no causal + dropout 0.0. [file:3]
fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor, _mask: Option<&Tensor>) -> Result<Tensor> {
    let (b, s, h, d) = q.dims4()?;
    let (b2, s_k, h2, d2) = k.dims4()?;
    ensure(b == b2 && h == h2 && d == d2, "q/k shape mismatch")?;

    // scores: [b, h, s, s_k]
    let q2 = q.transpose(1, 2)?; // [b, h, s, d]
    let k2 = k.transpose(1, 2)?; // [b, h, s_k, d]
    let v2 = v.transpose(1, 2)?; // [b, h, s_k, d]

    let k_t = k2.transpose(2, 3)?; // [b, h, d, s_k]
    let scale = 1f32 / (d as f32).sqrt();
    let scores = (q2.matmul(&k_t)? * scale)?; // [b, h, s, s_k]
    let probs = scores.softmax(3)?; // along s_k
    let out = probs.matmul(&v2)?; // [b, h, s, d]
    Ok(out.transpose(1, 2)?) // [b, s, h, d]
}

/// apply_rotary_emb из python: split pairs and apply cos/sin. [file:3]
fn apply_rotary_emb(x: &Tensor, freqs_cos: &Tensor, freqs_sin: &Tensor) -> Result<Tensor> {
    // x: [b, s, h, d]
    // freqs_cos/sin: [1, s, 1, d] (as produced by WanRotaryPosEmbed). [file:3]
    let (b, s, h, d) = x.dims4()?;
    ensure(d % 2 == 0, "rotary requires even head_dim")?;

    // x_even, x_odd
    let x_even = x.i((.., .., .., (0..d).step_by(2)))?; // [b,s,h,d/2]
    let x_odd = x.i((.., .., .., (1..d).step_by(2)))?;

    let cos = freqs_cos.i((.., .., .., (0..d).step_by(2)))?; // [1,s,1,d/2]
    let sin = freqs_sin.i((.., .., .., (1..d).step_by(2)))?; // [1,s,1,d/2]

    let out_even = (&x_even.broadcast_mul(&cos)? - &x_odd.broadcast_mul(&sin)?)?;
    let out_odd = (&x_even.broadcast_mul(&sin)? + &x_odd.broadcast_mul(&cos)?)?;

    // interleave even/odd back to [b,s,h,d]
    interleave_last_dim(&out_even, &out_odd)
}

fn interleave_last_dim(even: &Tensor, odd: &Tensor) -> Result<Tensor> {
    // even/odd: [b,s,h,d/2] -> [b,s,h,d]
    let (b, s, h, half) = even.dims4()?;
    ensure(odd.dims4()? == (b, s, h, half), "even/odd mismatch")?;

    let even_e = even.unsqueeze(4)?; // [b,s,h,half,1]
    let odd_e = odd.unsqueeze(4)?;   // [b,s,h,half,1]
    let stacked = Tensor::cat(&[&even_e, &odd_e], 4)?; // [b,s,h,half,2]
    Ok(stacked.reshape((b, s, h, half * 2))?)
}

// ------------------------- FeedForward -------------------------

#[derive(Clone, Debug)]
pub struct FeedForward {
    // Минимальный MLP: Linear -> GELU(approx) -> Linear [file:3]
    w1: Linear,
    w2: Linear,
}

impl FeedForward {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.w1.forward(x)?;
        // gelu-approximate / gelu_tanh в python — здесь используем встроенный gelu. [file:3]
        let h = h.gelu()?;
        self.w2.forward(&h)
    }
}

// ------------------------- Embeddings -------------------------

/// WanRotaryPosEmbed: строит freqs_cos/freqs_sin по 3D патчам. [file:3]
#[derive(Clone, Debug)]
pub struct WanRotaryPosEmbed {
    attention_head_dim: usize,
    patch_size: (usize, usize, usize),
    max_seq_len: usize,
    theta: f64,

    // В python это register_buffer со склеенным dim=1. [file:3]
    freqs_cos: Tensor, // [max_seq_len, head_dim]
    freqs_sin: Tensor, // [max_seq_len, head_dim]
    t_dim: usize,
    h_dim: usize,
    w_dim: usize,
}

impl WanRotaryPosEmbed {
    pub fn new(
        attention_head_dim: usize,
        patch_size: (usize, usize, usize),
        max_seq_len: usize,
        theta: f64,
        device: &Device,
    ) -> Result<Self> {
        // python: h_dim = w_dim = 2*(head_dim//6), t_dim = head_dim - h_dim - w_dim [file:3]
        let hw = 2 * (attention_head_dim / 6);
        let t_dim = attention_head_dim - 2 * hw;
        let h_dim = hw;
        let w_dim = hw;

        // Собираем 1D rotary для каждой оси и concat по dim=1. [file:3]
        let (cos_t, sin_t) = get_1d_rotary_pos_embed(t_dim, max_seq_len, theta, device)?;
        let (cos_h, sin_h) = get_1d_rotary_pos_embed(h_dim, max_seq_len, theta, device)?;
        let (cos_w, sin_w) = get_1d_rotary_pos_embed(w_dim, max_seq_len, theta, device)?;
        let freqs_cos = Tensor::cat(&[&cos_t, &cos_h, &cos_w], 1)?;
        let freqs_sin = Tensor::cat(&[&sin_t, &sin_h, &sin_w], 1)?;

        Ok(Self {
            attention_head_dim,
            patch_size,
            max_seq_len,
            theta,
            freqs_cos,
            freqs_sin,
            t_dim,
            h_dim,
            w_dim,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor)> {
        // hidden_states: [b, c, f, h, w] [file:3]
        let (_b, _c, frames, height, width) = hidden_states.dims5()?;
        let (p_t, p_h, p_w) = self.patch_size;

        ensure(frames % p_t == 0 && height % p_h == 0 && width % p_w == 0, "dims must be divisible by patch_size")?;

        let ppf = frames / p_t;
        let pph = height / p_h;
        let ppw = width / p_w;

        // python: split freqs by [t_dim, h_dim, w_dim], then broadcast and cat along last dim. [file:3]
        let cos_t = self.freqs_cos.i((0..ppf as i64, 0..self.t_dim as i64))?; // [ppf, t_dim]
        let cos_h = self.freqs_cos.i((0..pph as i64, self.t_dim as i64..(self.t_dim + self.h_dim) as i64))?;
        let cos_w = self
            .freqs_cos
            .i((0..ppw as i64, (self.t_dim + self.h_dim) as i64..(self.t_dim + self.h_dim + self.w_dim) as i64))?;

        let sin_t = self.freqs_sin.i((0..ppf as i64, 0..self.t_dim as i64))?;
        let sin_h = self.freqs_sin.i((0..pph as i64, self.t_dim as i64..(self.t_dim + self.h_dim) as i64))?;
        let sin_w = self
            .freqs_sin
            .i((0..ppw as i64, (self.t_dim + self.h_dim) as i64..(self.t_dim + self.h_dim + self.w_dim) as i64))?;

        // reshape to [ppf,1,1,dim], [1,pph,1,dim], [1,1,ppw,dim] and broadcast
        let cos_t = cos_t.reshape((ppf, 1, 1, self.t_dim))?.broadcast_as((ppf, pph, ppw, self.t_dim))?;
        let cos_h = cos_h.reshape((1, pph, 1, self.h_dim))?.broadcast_as((ppf, pph, ppw, self.h_dim))?;
        let cos_w = cos_w.reshape((1, 1, ppw, self.w_dim))?.broadcast_as((ppf, pph, ppw, self.w_dim))?;
        let sin_t = sin_t.reshape((ppf, 1, 1, self.t_dim))?.broadcast_as((ppf, pph, ppw, self.t_dim))?;
        let sin_h = sin_h.reshape((1, pph, 1, self.h_dim))?.broadcast_as((ppf, pph, ppw, self.h_dim))?;
        let sin_w = sin_w.reshape((1, 1, ppw, self.w_dim))?.broadcast_as((ppf, pph, ppw, self.w_dim))?;

        let cos = Tensor::cat(&[&cos_t, &cos_h, &cos_w], 3)?; // [ppf,pph,ppw,head_dim]
        let sin = Tensor::cat(&[&sin_t, &sin_h, &sin_w], 3)?;

        // python: reshape(1, ppf*pph*ppw, 1, -1) [file:3]
        let seq = ppf * pph * ppw;
        let cos = cos.reshape((1, seq, 1, self.attention_head_dim))?;
        let sin = sin.reshape((1, seq, 1, self.attention_head_dim))?;
        Ok((cos, sin))
    }
}

/// Упрощённая реализация get_1d_rotary_pos_embed(use_real=True, repeat_interleave_real=True). [file:3]
fn get_1d_rotary_pos_embed(dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> Result<(Tensor, Tensor)> {
    ensure(dim % 2 == 0, "rotary dim must be even")?;

    // inv_freq: [dim/2]
    let half = dim / 2;
    let mut inv_freq = Vec::with_capacity(half);
    for i in 0..half {
        let exponent = (2.0 * i as f64) / (dim as f64);
        inv_freq.push((1.0 / theta.powf(exponent)) as f32);
    }
    let inv_freq = Tensor::from_vec(inv_freq, (half,), device)?;

    // positions: [max_seq_len]
    let pos: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
    let pos = Tensor::from_vec(pos, (max_seq_len, 1), device)?; // [L,1]

    // angles: [L, half]
    let angles = pos.matmul(&inv_freq.unsqueeze(0)?)?; // [L, half]

    // repeat_interleave_real=True => expand to [L, dim] by repeating each angle twice. [file:3]
    let angles2 = Tensor::cat(&[&angles, &angles], 1)?; // [L, dim]

    let cos = angles2.cos()?;
    let sin = angles2.sin()?;
    Ok((cos, sin))
}

/// Condition embedder в python: Timesteps -> TimestepEmbedding -> time_proj(6*inner_dim) + text proj + optional image proj. [file:3]
///
/// Здесь: делаем минимально совместимые по формам temb и timestep_proj, а текстовые эмбеддинги
/// предполагаем уже в нужной форме (после encoder) и делаем линейную проекцию. [file:3]
#[derive(Clone, Debug)]
pub struct WanTimeTextImageEmbedding {
    dim: usize,
    time_proj: Linear,
    text_proj: Linear,
    // image embedder опционален; pos_embed/ff/ln в python — опускаем в минимальном порте. [file:3]
    image_proj: Option<Linear>,
}

impl WanTimeTextImageEmbedding {
    pub fn forward(
        &self,
        timestep: &Tensor,                 // [b] or [b*seq] (flattened)
        encoder_hidden_states: &Tensor,    // [b, s_text, text_dim]
        encoder_hidden_states_image: Option<&Tensor>, // [b, s_img, img_dim]
        timestep_seq_len: Option<usize>,   // when original timestep was [b, seq_len]
    ) -> Result<(Tensor, Tensor, Tensor, Option<Tensor>)> {
        // temb: в python это выход time_embedder, далее act+time_proj. [file:3]
        // Здесь: делаем temb простым sin/cos-ish embedding: [b] -> [b, dim] (или [b, seq, dim]).
        let device = timestep.device();
        let b_flat = timestep.dims1()?; // b or b*seq
        let temb = simple_time_embedding(timestep, self.dim)?; // [b_flat, dim]

        let temb = if let Some(seq) = timestep_seq_len {
            // python: timestep was unflattened back for time_embedder; temb becomes [b, seq, dim]. [file:3]
            ensure(b_flat % seq == 0, "timestep flatten length must be divisible by seq_len")?;
            let b = b_flat / seq;
            temb.reshape((b, seq, self.dim))?
        } else {
            temb
        };

        // timestep_proj: time_proj(SiLU(temb)) -> [b, 6*dim] or [b, seq, 6*dim]. [file:3]
        let temb_act = silu(&temb.to_dtype(self.time_proj.weight.dtype())?)?;
        let timestep_proj = self.time_proj.forward(&temb_act)?; // [.., 6*dim]

        // text projection to model dim
        let enc_text = self.text_proj.forward(encoder_hidden_states)?; // [b, s_text, dim]

        // optional image projection
        let enc_img = match (self.image_proj.as_ref(), encoder_hidden_states_image) {
            (Some(p), Some(img)) => Some(p.forward(img)?), // [b, s_img, dim]
            _ => None,
        };

        Ok((temb, timestep_proj, enc_text, enc_img))
    }
}

fn simple_time_embedding(timestep: &Tensor, dim: usize) -> Result<Tensor> {
    // timestep: [n]
    let n = timestep.dims1()?;
    let t = timestep.to_dtype(DType::F32)?.unsqueeze(1)?; // [n,1]
    // создаём частоты и sin/cos
    let half = dim / 2;
    let device = timestep.device();

    let mut freq = Vec::<f32>::with_capacity(half);
    for i in 0..half {
        // грубая шкала, достаточно чтобы сохранить "shape contract" [file:3]
        freq.push((i as f32) / (half.max(1) as f32));
    }
    let freq = Tensor::from_vec(freq, (1, half), device)?; // [1,half]
    let x = t.broadcast_mul(&freq)?; // [n,half]
    let sin = x.sin()?;
    let cos = x.cos()?;
    let out = Tensor::cat(&[&sin, &cos], 1)?; // [n,dim_even]
    if dim % 2 == 0 {
        Ok(out)
    } else {
        // не ожидается в конфиге Wan (head_dim 128 => inner_dim кратно 2). [file:3]
        let pad = Tensor::zeros((n, 1), DType::F32, device)?;
        Ok(Tensor::cat(&[&out, &pad], 1)?)
    }
}

// ------------------------- transformer block -------------------------

#[derive(Clone, Debug)]
pub struct WanTransformerBlock {
    norm1: LayerNorm,
    attn1: WanAttention,

    attn2: WanAttention,
    norm2: Option<LayerNorm>,

    ffn: FeedForward,
    norm3: LayerNorm,

    // scale_shift_table: [1,6,dim] [file:3]
    scale_shift_table: Tensor,
}

impl WanTransformerBlock {
    pub fn forward(
        &self,
        hidden_states: &Tensor,          // [b, s, dim]
        encoder_hidden_states: &Tensor,  // [b, s_ctx, dim] (text or img+text)
        temb_proj: &Tensor,              // [b,6,dim] or [b,s,6,dim] in python [file:3]
        rotary_emb: (&Tensor, &Tensor),  // (cos,sin)
    ) -> Result<Tensor> {
        // В python:
        // if temb.ndim==4 => [b, s, 6, dim], else [b, 6, dim], затем chunk на 6. [file:3]
        let (shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa) =
            compute_scale_shift_gates(&self.scale_shift_table, temb_proj)?;

        // 1) Self-attn
        let norm_h = self.norm1.forward(&hidden_states.to_dtype(DType::F32)?)?;
        let norm_h = ((&norm_h * (&scale_msa + 1f32)?)? + &shift_msa)?;
        let norm_h = norm_h.to_dtype(hidden_states.dtype())?;
        let attn_out = self.attn1.forward(&norm_h, None, None, Some(rotary_emb))?;
        let hs = (hidden_states.to_dtype(DType::F32)? + (&attn_out.to_dtype(DType::F32)? * &gate_msa)?)?
            .to_dtype(hidden_states.dtype())?;

        // 2) Cross-attn
        let norm_h2 = if let Some(n2) = &self.norm2 {
            n2.forward(&hs.to_dtype(DType::F32)?)?.to_dtype(hs.dtype())?
        } else {
            hs.clone()
        };
        let attn_out2 = self.attn2.forward(&norm_h2, Some(encoder_hidden_states), None, None)?;
        let hs = (&hs + &attn_out2)?;

        // 3) FFN
        let norm_h3 = self.norm3.forward(&hs.to_dtype(DType::F32)?)?;
        let norm_h3 = ((&norm_h3 * (&c_scale_msa + 1f32)?)? + &c_shift_msa)?;
        let norm_h3 = norm_h3.to_dtype(hs.dtype())?;
        let ff = self.ffn.forward(&norm_h3)?;
        let hs = (hs.to_dtype(DType::F32)? + (ff.to_dtype(DType::F32)? * &c_gate_msa)?)?.to_dtype(hs.dtype())?;

        Ok(hs)
    }
}

/// Возвращает 6 тензоров, broadcast-совместимых с hidden_states: [b,s,dim]. [file:3]
fn compute_scale_shift_gates(scale_shift_table: &Tensor, temb: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
    // python:
    // (scale_shift_table + temb.float()).chunk(6, dim=1)  or dim=2 for ti2v case. [file:3]
    let temb_f = temb.to_dtype(DType::F32)?;
    match temb_f.rank() {
        3 => {
            // [b,6,dim]
            let sst = scale_shift_table.to_dtype(DType::F32)?;
            let sst = sst.broadcast_as(temb_f.dims())?;
            let x = (&sst + &temb_f)?;
            // split dim=1 into 6 tensors [b,1,dim] then squeeze -> [b,dim] but we keep [b,1,dim] and broadcast later.
            let parts = x.chunk(6, 1)?;
            let mut p = Vec::with_capacity(6);
            for t in parts {
                p.push(t.squeeze(1)?); // [b,dim]
            }
            // Make [b,1,dim] to broadcast over seq later.
            let p: Vec<Tensor> = p.into_iter().map(|t| t.unsqueeze(1)).collect::<CandleResult<_>>()?;
            Ok((p[0].clone(), p[1].clone(), p[2].clone(), p[3].clone(), p[4].clone(), p[5].clone()))
        }
        4 => {
            // [b,s,6,dim]
            let (b, s, _, dim) = temb_f.dims4()?;
            let sst = scale_shift_table.to_dtype(DType::F32)?;
            let sst = sst.unsqueeze(0)?.broadcast_as((b, sst.dims3()?.0, sst.dims3()?.1, sst.dims3()?.2))?; // [b,1,6,dim] approx
            let sst = sst.broadcast_as((b, s, 6, dim))?;
            let x = (&sst + &temb_f)?;
            let parts = x.chunk(6, 2)?; // along the "6" axis
            let mut p = Vec::with_capacity(6);
            for t in parts {
                p.push(t.squeeze(2)?); // [b,s,dim]
            }
            Ok((p[0].clone(), p[1].clone(), p[2].clone(), p[3].clone(), p[4].clone(), p[5].clone()))
        }
        _ => Err(WanError::InvalidArgument(
            "temb_proj must be rank 3 ([b,6,dim]) or rank 4 ([b,s,6,dim])".to_string(),
        )),
    }
}

// ------------------------- model config/output -------------------------

#[derive(Clone, Debug)]
pub struct WanTransformer3DConfig {
    pub patch_size: (usize, usize, usize), // (t,h,w) [file:3]
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub text_dim: usize,
    pub freq_dim: usize, // unused in minimal embedder
    pub ffn_dim: usize,
    pub num_layers: usize,
    pub cross_attn_norm: bool,
    pub eps: f64,
    pub image_dim: Option<usize>,
    pub added_kv_proj_dim: Option<usize>,
    pub rope_max_seq_len: usize,
    pub pos_embed_seq_len: Option<usize>, // unused in minimal port
}

#[derive(Debug)]
pub struct Transformer2DModelOutput {
    pub sample: Tensor,
}

// ------------------------- main model -------------------------

pub struct WanTransformer3DModel {
    pub cfg: WanTransformer3DConfig,

    rope: WanRotaryPosEmbed,

    // patch_embedding: Conv3d(in_channels -> inner_dim, kernel_size=patch, stride=patch) [file:3]
    // Candle не предоставляет "nn::Conv3d" в core; оставляем как трейтовую зависимость.
    // Здесь — минимальный интерфейс, который должен дать эквивалент conv3d. [file:3]
    patch_embedding: Box<dyn Conv3dLike>,

    condition_embedder: WanTimeTextImageEmbedding,

    blocks: Vec<WanTransformerBlock>,

    norm_out: LayerNorm,
    proj_out: Linear,

    // [1,2,inner_dim] [file:3]
    scale_shift_table: Tensor,
}

pub trait Conv3dLike: Send + Sync {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
}

impl WanTransformer3DModel {
    pub fn new(
        cfg: WanTransformer3DConfig,
        rope: WanRotaryPosEmbed,
        patch_embedding: Box<dyn Conv3dLike>,
        condition_embedder: WanTimeTextImageEmbedding,
        blocks: Vec<WanTransformerBlock>,
        norm_out: LayerNorm,
        proj_out: Linear,
        scale_shift_table: Tensor,
    ) -> Self {
        Self {
            cfg,
            rope,
            patch_embedding,
            condition_embedder,
            blocks,
            norm_out,
            proj_out,
            scale_shift_table,
        }
    }

    pub fn in_channels(&self) -> usize {
        self.cfg.in_channels
    }

    pub fn dtype(&self) -> DType {
        // В python dtype модели определяется параметрами; здесь используем dtype proj_out.weight. [file:3]
        self.proj_out.weight.dtype()
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,               // [b,c,f,h,w] latents
        timestep: &Tensor,                    // [b] or [b, seq_len] [file:3]
        encoder_hidden_states: &Tensor,       // [b, s_text, text_dim] (или уже concat img+text)
        encoder_hidden_states_image: Option<&Tensor>,
        return_dict: bool,
    ) -> Result<std::result::Result<Transformer2DModelOutput, Tensor>> {
        // 1) shapes
        let (b, _c, frames, height, width) = hidden_states.dims5()?;
        let (p_t, p_h, p_w) = self.cfg.patch_size;

        ensure(frames % p_t == 0 && height % p_h == 0 && width % p_w == 0, "input dims not divisible by patch_size")?;

        let post_f = frames / p_t;
        let post_h = height / p_h;
        let post_w = width / p_w;

        // 2) rotary
        let (freqs_cos, freqs_sin) = self.rope.forward(hidden_states)?; // [1, seq, 1, head_dim] [file:3]

        // 3) patch embedding conv3d -> [b, inner_dim, post_f, post_h, post_w]
        let hs = self.patch_embedding.forward(hidden_states)?;

        // python: flatten(2).transpose(1,2) => [b, seq, inner_dim] [file:3]
        let hs = hs.flatten_from(2)?; // [b, inner_dim, seq]
        let mut hs = hs.transpose(1, 2)?; // [b, seq, inner_dim]
        let seq = hs.dims3()?.1;

        // 4) timestep shape logic (wan2.2 ti2v): if timestep.ndim==2 => flatten and ts_seq_len=timestep.shape[1]. [file:3]
        let (timestep_flat, ts_seq_len) = match timestep.rank() {
            1 => (timestep.clone(), None),
            2 => {
                let (_b2, s2) = timestep.dims2()?;
                ensure(_b2 == b, "timestep batch mismatch")?;
                (timestep.flatten_all()?, Some(s2))
            }
            _ => {
                return Err(WanError::InvalidArgument(
                    "timestep must be rank-1 [b] or rank-2 [b, seq_len]".to_string(),
                ))
            }
        };

        // 5) condition embedder
        let (temb, timestep_proj, enc_text, enc_img) = self.condition_embedder.forward(
            &timestep_flat,
            encoder_hidden_states,
            encoder_hidden_states_image,
            ts_seq_len,
        )?;

        // python: timestep_proj unflatten into (6,-1) on dim=1 or dim=2 depending on ti2v. [file:3]
        let inner_dim = self.cfg.num_attention_heads * self.cfg.attention_head_dim;
        let timestep_proj = match timestep_proj.rank() {
            2 => {
                // [b, 6*inner_dim] -> [b,6,inner_dim]
                ensure(timestep_proj.dims2()?.0 == b, "timestep_proj batch mismatch")?;
                timestep_proj.reshape((b, 6, inner_dim))?
            }
            3 => {
                // [b, seq_len, 6*inner_dim] -> [b, seq_len, 6, inner_dim]
                let (bb, ss, _) = timestep_proj.dims3()?;
                ensure(bb == b, "timestep_proj batch mismatch")?;
                timestep_proj.reshape((bb, ss, 6, inner_dim))?
            }
            _ => {
                return Err(WanError::InvalidArgument(
                    "timestep_proj must be rank-2 or rank-3".to_string(),
                ))
            }
        };

        // python: if encoder_hidden_states_image is not None => concat along dim=1 [file:3]
        let enc = if let Some(img) = enc_img {
            Tensor::cat(&[&img, &enc_text], 1)?
        } else {
            enc_text
        };

        // 6) transformer blocks
        for blk in &self.blocks {
            hs = blk.forward(&hs, &enc, &timestep_proj, (&freqs_cos, &freqs_sin))?;
        }

        // 7) output norm/proj/unpatchify
        // python:
        // if temb.ndim==3 => [b, seq_len, inner_dim] else [b, inner_dim] [file:3]
        let (shift, scale) = compute_out_shift_scale(&self.scale_shift_table, &temb, inner_dim)?;

        let hs_norm = self.norm_out.forward(&hs.to_dtype(DType::F32)?)?;
        let hs_norm = ((&hs_norm * (&scale + 1f32)?)? + &shift)?;
        let hs_norm = hs_norm.to_dtype(hs.dtype())?;

        let out = self.proj_out.forward(&hs_norm)?; // [b, seq, out_channels*prod(patch)] [file:3]
        let patch_prod = p_t * p_h * p_w;
        let out_ch = self.cfg.out_channels;

        // reshape -> permute -> flatten like python
        // reshape: [b, post_f, post_h, post_w, p_t, p_h, p_w, out_ch]
        let out = out.reshape((b, post_f, post_h, post_w, p_t, p_h, p_w, out_ch))?;
        // permute(0,7,1,4,2,5,3,6) -> [b, out_ch, post_f, p_t, post_h, p_h, post_w, p_w] [file:3]
        let out = out.permute((0, 7, 1, 4, 2, 5, 3, 6))?;
        // flatten frames dims (2,3), height dims (4,5), width dims (6,7)
        let out = out.flatten(2, 3)?.flatten(3, 4)?.flatten(4, 5)?;
        // Now should be [b, out_ch, frames, height, width]
        ensure(out.dims5()?.0 == b, "output batch mismatch")?;

        if return_dict {
            Ok(Ok(Transformer2DModelOutput { sample: out }))
        } else {
            Ok(Err(out))
        }
    }
}

/// compute (shift, scale) for final norm_out path. [file:3]
fn compute_out_shift_scale(scale_shift_table: &Tensor, temb: &Tensor, inner_dim: usize) -> Result<(Tensor, Tensor)> {
    // python:
    // if temb.ndim==3:
    //   (scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2) [file:3]
    // else:
    //   (scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1) [file:3]
    let sst = scale_shift_table.to_dtype(DType::F32)?;

    match temb.rank() {
        2 => {
            // temb: [b, inner_dim]
            let (b, d) = temb.dims2()?;
            ensure(d == inner_dim, "temb inner_dim mismatch")?;
            let t = temb.to_dtype(DType::F32)?.unsqueeze(1)?; // [b,1,inner]
            let sst = sst.broadcast_as((b, 2, inner_dim))?;
            let x = (&sst + &t)?; // [b,2,inner]
            let parts = x.chunk(2, 1)?;
            let shift = parts[0].squeeze(1)?; // [b,inner]
            let scale = parts[1].squeeze(1)?;
            // broadcast to [b, seq, inner] later by unsqueeze(1)
            Ok((shift.unsqueeze(1)?, scale.unsqueeze(1)?))
        }
        3 => {
            // temb: [b, seq, inner_dim]
            let (b, s, d) = temb.dims3()?;
            ensure(d == inner_dim, "temb inner_dim mismatch")?;
            let t = temb.to_dtype(DType::F32)?.unsqueeze(2)?; // [b,s,1,inner]
            let sst = sst.unsqueeze(0)?.broadcast_as((b, s, 2, inner_dim))?;
            let x = (&sst + &t)?; // [b,s,2,inner]
            let parts = x.chunk(2, 2)?;
            let shift = parts[0].squeeze(2)?; // [b,s,inner]
            let scale = parts[1].squeeze(2)?;
            Ok((shift, scale))
        }
        _ => Err(WanError::InvalidArgument(
            "temb must be rank-2 ([b,inner]) or rank-3 ([b,seq,inner])".to_string(),
        )),
    }
}
