# Единый отчёт: ревью + дебаг

## 1) Краткий вердикт

- **Общий риск: HIGH** — Обнаружены критические расхождения с референсом diffusers, которые гарантированно приводят к некорректной работе модели.
- **Готовность к мерджу: НЕТ** — Требуется исправление как минимум 3-4 blocker-уровня багов.

**Ключевые причины:**
1. **FPS-1 не применяется** — SVD обучен на fps-1, но в Rust передаётся fps как есть
2. **Формат latents отличается** — diffusers использует [B, F, C, H, W], Rust использует [B*F, C, H, W]
3. **Guidance scale применяется по-разному** — diffusers использует per-frame guidance, Rust использует скаляр
4. **Отсутствует decode_chunk_size** — Rust декодирует все фреймы сразу, что может вызвать OOM

## 2) Что было проверено

### Проверено:
| Файл | Модуль | Описание |
|------|--------|----------|
| `src/svd/pipeline.rs` | Pipeline | Основной пайплайн генерации |
| `src/svd/scheduler.rs` | Scheduler | Euler Discrete Scheduler |
| `src/svd/unet/model.rs` | UNet | Основная модель UNet |
| `src/svd/unet/blocks.rs` | Blocks | Down/Mid/Up блоки |
| `src/svd/unet/resnet.rs` | ResNet | Spatio-Temporal ResNet блоки |
| `src/svd/unet/transformer.rs` | Transformer | Spatio-Temporal Transformer |
| `src/svd/vae/mod.rs` | VAE | AutoencoderKLTemporalDecoder |
| `src/svd/vae/decoder.rs` | Decoder | Temporal Decoder |
| `src/svd/clip.rs` | CLIP | Vision encoder |
| `src/svd/config.rs` | Config | Конфигурации |
| `src/bin/svd.rs` | CLI | Точка входа |

### Референс (diffusers):
| Файл | Описание |
|------|----------|
| `tp/diffusers/.../pipeline_stable_video_diffusion.py` | Референсный пайплайн |
| `tp/diffusers/.../unet_spatio_temporal_condition.py` | Референсный UNet |
| `tp/diffusers/.../scheduling_euler_discrete.py` | Референсный Scheduler |

### НЕ проверено (не хватает данных):
- Логи запуска с ошибкой
- Stack trace при падении
- Конфигурация конкретной модели (scheduler_config.json, unet/config.json)
- CUDA device info / memory usage

## 3) Находки ревью (качество/архитектура)

### [Severity: Blocker] 🔴 FPS conditioning: отсутствует fps-1

**Доказательство:**
- diffusers `pipeline_stable_video_diffusion.py:507`:
  ```python
  # NOTE: Stable Video Diffusion was conditioned on fps - 1, which is why it is reduced here.
  # See: https://github.com/Stability-AI/generative-models/blob/...
  fps = fps - 1
  ```
- Rust `src/svd/pipeline.rs:113`:
  ```rust
  config.fps as f32,  // ← fps передаётся как есть!
  ```

**Почему это важно:** SVD модель обучена с fps-1 conditioning. Передача неправильного значения полностью нарушает time embedding и генерацию.

**Рекомендация:** Изменить строку 113 на `(config.fps - 1) as f32`.

---

### [Severity: Blocker] 🔴 Формат latents: [B*F, C, H, W] vs [B, F, C, H, W]

**Доказательство:**
- diffusers `pipeline_stable_video_diffusion.py:345-351`:
  ```python
  shape = (
      batch_size,
      num_frames,
      num_channels_latents // 2,  # 4 канала
      height // self.vae_scale_factor,
      width // self.vae_scale_factor,
  )
  ```
- diffusers `unet_spatio_temporal_condition.py:350`:
  ```python
  batch_size, num_frames = sample.shape[:2]  # ожидает 5D tensor!
  ```
- Rust `src/svd/pipeline.rs:102-108`:
  ```rust
  let latents = Tensor::randn(
      0f32,
      1f32,
      (batch_size * num_frames, 4, latent_height, latent_width),  // ← 4D tensor!
      ...
  )?
  ```

**Почему это важно:** UNet в diffusers ожидает 5D тензор [B, F, C, H, W] и сам делает flatten внутри forward. В Rust мы уже передаём flattened tensor, что нарушает логику embeddings repeat.

**Рекомендация:** Изменить shape latents на `(batch_size, num_frames, 4, latent_height, latent_width)` и убедиться, что UNet корректно обрабатывает 5D input.

---

### [Severity: Blocker] 🔴 Guidance scale: скаляр vs per-frame

**Доказательство:**
- diffusers `pipeline_stable_video_diffusion.py:564-567`:
  ```python
  guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
  guidance_scale = guidance_scale.to(device, latents.dtype)
  guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
  guidance_scale = _append_dims(guidance_scale, latents.ndim)  # [B, F, 1, 1, 1]
  ```
- Rust `src/svd/pipeline.rs:136-138`:
  ```rust
  let guidance_scale = config.min_guidance_scale
      + (config.max_guidance_scale - config.min_guidance_scale)
          * (i as f64 / (config.num_inference_steps - 1) as f64);  // ← скаляр на весь batch!
  ```

**Почему это важно:** В diffusers guidance scale интерполируется **по фреймам**, а не по шагам инференса. Это критически важно для качественной генерации — первые фреймы получают min_guidance, последние — max_guidance.

**Рекомендация:** Создать тензор guidance_scale shape [1, F, 1, 1, 1] и умножать каждый фрейм на соответствующее значение.

---

### [Severity: Blocker] 🔴 Image latents concatenation: по frames vs по channels

**Доказательство:**
- diffusers `pipeline_stable_video_diffusion.py:581`:
  ```python
  latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)  # dim=2 это channels в 5D
  ```
- Rust `src/svd/pipeline.rs:167`:
  ```rust
  let latent_input_cond = Tensor::cat(&[&latent_model_input, &image_cond_latents], 1)?;  // dim=1 в 4D
  ```

**Почему это важно:** В diffusers latents имеют формат [B, F, C, H, W] и concat идёт по dim=2 (channels). В Rust latents [B*F, C, H, W] и concat по dim=1 (channels). Это может быть эквивалентно, но требует проверки shape.

**Рекомендация:** Проверить и привести к единому формату с diffusers.

---

### [Severity: Major] 🟠 Batched CFG vs раздельные forward passes

**Доказательство:**
- diffusers `pipeline_stable_video_diffusion.py:577-578`:
  ```python
  latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
  latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
  ```
  Затем один forward pass с batch=2.
  
- Rust `src/svd/pipeline.rs:150-189`:
  ```rust
  if do_classifier_free_guidance {
      // Два ОТДЕЛЬНЫХ forward pass!
      let noise_pred_uncond = self.unet.forward(...)?;
      let noise_pred_cond = self.unet.forward(...)?;
  }
  ```

**Почему это важно:** 
1. Два отдельных forward pass vs один batched — разная производительность
2. Если есть BatchNorm или другие batch-sensitive операции, результаты могут отличаться

**Рекомендация:** Объединить в один batched forward pass как в diffusers.

---

### [Severity: Major] 🟠 Отсутствует decode_chunk_size

**Доказательство:**
- diffusers `pipeline_stable_video_diffusion.py:290-310`:
  ```python
  def decode_latents(self, latents, num_frames, decode_chunk_size=14):
      for i in range(0, latents.shape[0], decode_chunk_size):
          num_frames_in = latents[i : i + decode_chunk_size].shape[0]
          frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
          frames.append(frame)
  ```
- Rust `src/svd/pipeline.rs:197`:
  ```rust
  let video_frames = self.vae.decode(&latents, num_frames)?;  // Все фреймы сразу
  ```

**Почему это важно:** Декодирование всех 14 фреймов сразу требует значительно больше VRAM. При 576x1024 это может вызвать OOM на картах с <12GB.

**Рекомендация:** Реализовать chunked decode как в diffusers.

---

### [Severity: Major] 🟠 UNet forward: emb не repeat_interleave по фреймам

**Доказательство:**
- diffusers `unet_spatio_temporal_condition.py:373-377`:
  ```python
  emb = emb.repeat_interleave(num_frames, dim=0, output_size=emb.shape[0] * num_frames)
  encoder_hidden_states = encoder_hidden_states.repeat_interleave(
      num_frames, dim=0, output_size=encoder_hidden_states.shape[0] * num_frames
  )
  ```
- Rust `src/svd/unet/model.rs:310-311`:
  ```rust
  let emb = (t_emb + aug_emb)?;  // emb уже [B*F], но вопрос в том как это получено
  ```

**Почему это важно:** В diffusers emb рассчитывается для batch [B], а затем repeat_interleave для [B*F]. В Rust timestep уже передаётся как [B*F], поэтому emb тоже [B*F]. Это может быть корректно, но требует проверки.

---

### [Severity: Minor] 🟡 Noise augmentation: порядок операций

**Доказательство:**
- diffusers `pipeline_stable_video_diffusion.py:511-512`:
  ```python
  noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
  image = image + noise_aug_strength * noise
  ```
  Добавляют шум к уже препроцессированному image ПЕРЕД encode.
  
- Rust `src/svd/pipeline.rs:91-93`:
  ```rust
  let noise = image_latents.randn_like(0.0, noise_aug_strength)?;
  let image_latents_augmented = (&image_latents + noise)?;
  ```
  Добавляют шум к latents ПОСЛЕ encode.

**Почему это важно:** В diffusers шум добавляется к изображению в pixel space, затем идёт encode. В Rust шум добавляется уже в latent space. Это может дать разные результаты.

**Рекомендация:** Добавлять шум до encode, как в diffusers.

---

### [Severity: Minor] 🟡 Timestep type: continuous handling

**Доказательство:**
- diffusers `scheduling_euler_discrete.py:254-257`:
  ```python
  if timestep_type == "continuous" and prediction_type == "v_prediction":
      self.timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas])
  ```
  Timesteps это **0.25 * log(sigma)**, передаётся в UNet.

- Rust config `src/svd/config.rs:102`:
  ```rust
  timestep_type: "continuous".to_string(),
  ```
- Rust scheduler `src/svd/scheduler.rs:151-156`:
  ```rust
  self.timesteps = if self.config.timestep_type == "continuous" {
      step_sigmas[..step_sigmas.len() - 1]
          .iter()
          .map(|&s| s.ln().neg())  // ← -ln(sigma), а не 0.25 * ln(sigma)!
          .collect()
  }
  ```

**Почему это важно:** Коэффициент 0.25 отсутствует в Rust реализации.

**Рекомендация:** Изменить на `s.ln() * 0.25` или `-s.ln() * 0.25` (проверить знак).

---

### [Severity: Nit] 🔵 CLIP image preprocessing

**Доказательство:**
- diffusers использует antialias resize + specific CLIP normalization
- Rust `src/svd/pipeline.rs:76`:
  ```rust
  let clip_image = image.upsample_nearest2d(224, 224)?;  // nearest neighbor, не антиалиас!
  ```

**Рекомендация:** Использовать bilinear/bicubic интерполяцию с anti-aliasing как в diffusers.

## 4) Баги и первопричины (debug)

| ID | Симптом | Вероятная первопричина | Доказательства | Уверенность | Исправление | Как проверить |
|----|---------|------------------------|---------------|-------------|--------------|---------------|
| BUG-1 | Прерывание после начала генерации | OOM из-за отсутствия chunked decode или неправильного формата тензоров | VAE decode всех фреймов сразу, diffusers использует chunks | 70% | Реализовать decode_chunk_size | nvidia-smi мониторинг VRAM |
| BUG-2 | Некорректные фреймы (если доходит) | Неправильный fps conditioning (fps вместо fps-1) | См. доказательства в п.3 | 95% | Исправить fps на fps-1 | Сравнить output с diffusers |
| BUG-3 | Некорректные фреймы | Guidance scale per-step вместо per-frame | См. доказательства в п.3 | 90% | Исправить guidance scale | Сравнить intermediate latents |
| BUG-4 | Некорректные фреймы | Noise augmentation в latent space вместо pixel space | См. доказательства в п.3 | 80% | Переместить noise aug до encode | Сравнить image_latents |
| BUG-5 | Тихий crash или неверные результаты | Timestep scaling 0.25 отсутствует | См. доказательства в п.3 | 75% | Добавить коэффициент | Сравнить timesteps |

## 5) Предложение фикса (минимальный план)

### Шаг 1: FPS conditioning (критично)
```rust
// src/svd/pipeline.rs:113
- config.fps as f32,
+ (config.fps.saturating_sub(1)) as f32,  // SVD conditioned on fps-1
```

### Шаг 2: Guidance scale per-frame (критично)
```rust
// src/svd/pipeline.rs, перед denoising loop
// Создать per-frame guidance scale tensor
let guidance_scales: Vec<f64> = (0..num_frames)
    .map(|f| {
        config.min_guidance_scale
            + (config.max_guidance_scale - config.min_guidance_scale)
                * (f as f64 / (num_frames - 1).max(1) as f64)
    })
    .collect();
// В CFG: умножать на соответствующий элемент для каждого фрейма
```

### Шаг 3: Noise augmentation (важно)
```rust
// src/svd/pipeline.rs:87-93
// Перенести noise aug ПЕРЕД encode
pub fn generate(&mut self, image: &Tensor, config: &SvdInferenceConfig) -> Result<Tensor> {
    // ...
    
    // Add noise augmentation to image BEFORE encoding (как в diffusers)
    let noise_aug_strength = config.noise_aug_strength;
    let noise = image.randn_like(0.0, 1.0)?;
    let image_augmented = (image + &(noise * noise_aug_strength)?)?;
    
    // Encode augmented image
    let image_latents = self.vae.encode_to_latent(&image_augmented)?;
```

### Шаг 4: Timestep scaling (важно)
```rust
// src/svd/scheduler.rs:155
- .map(|&s| s.ln().neg())
+ .map(|&s| s.ln() * 0.25)  // diffusers: 0.25 * sigma.log()
```

### Шаг 5: Decode chunking (для стабильности)
```rust
// src/svd/vae/mod.rs:101
pub fn decode(&self, z: &Tensor, num_frames: usize, chunk_size: Option<usize>) -> Result<Tensor> {
    let chunk_size = chunk_size.unwrap_or(num_frames);
    let mut frames = Vec::new();
    
    for i in (0..num_frames).step_by(chunk_size) {
        let end = std::cmp::min(i + chunk_size, num_frames);
        let chunk = z.narrow(0, i, end - i)?;
        let decoded = self.temporal_decoder.forward(&chunk, ...)?;
        frames.push(decoded);
    }
    
    Tensor::cat(&frames, 0)
}
```

## 6) Тесты и проверки

### Юнит-тесты (добавить):
1. `test_fps_minus_one_conditioning` — проверить что fps-1 передаётся в added_time_ids
2. `test_per_frame_guidance_scale` — проверить shape и значения guidance_scale
3. `test_noise_augmentation_before_encode` — проверить что шум добавляется в pixel space
4. `test_timestep_scaling` — сравнить timesteps с diffusers

### Интеграционные тесты:
1. Сравнение image_embeddings с diffusers (численное)
2. Сравнение image_latents с diffusers после encode
3. Сравнение noise_pred на каждом шаге
4. Сравнение final latents

### Команды проверки:
```bash
# Запуск diffusers reference
python scripts/run_svd_diffusers.py

# Запуск Rust с отладкой
RUST_BACKTRACE=1 cargo run --bin svd -- --image test.png --model models/svd --steps 2

# Сравнение tensors
python scripts/compare_tensors.py output/rust output/diffusers
```

## 7) Риски и откаты

### Риски регрессий:
1. **Изменение fps conditioning** — может сломать существующие результаты если кто-то адаптировался
2. **Изменение guidance scale** — критически меняет quality/motion trade-off
3. **Изменение latent format** — самое опасное, может сломать весь pipeline

### План отката:
1. Создать feature flag `USE_DIFFUSERS_COMPAT` для переключения поведения
2. Сохранить старую логику под `#[cfg(feature = "legacy")]`
3. Версионировать config (v1 = legacy, v2 = diffusers_compat)

### Критичные места:
- `pipeline.rs:generate()` — основная логика
- `scheduler.rs:set_timesteps()` — timestep generation
- `vae/mod.rs:decode()` — decode chunking

## 8) Вопросы к автору / что нужно предоставить

1. **Логи/traceback** при crash — точное сообщение об ошибке
2. **nvidia-smi** вывод во время запуска — для понимания OOM ли это
3. **RUST_BACKTRACE=1** вывод — для понимания где именно crash
4. **Intermediate tensors** — сохранить latents после каждого шага для сравнения
5. **Модель config** — `scheduler/scheduler_config.json` и `unet/config.json` для проверки параметров
6. **Версия CUDA/cuDNN** — для воспроизведения
7. **Точная команда запуска** — параметры CLI

---

**Дата отчёта:** 2025-12-30  
**Ревьюер:** Claude Opus 4.5 (Code Reviewer + Debugger)
