</p>
<p align="left">
  <a href="README.md"><img src="https://img.shields.io/badge/English-232323" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-D65C5C" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-232323" alt="Português"></a>
</p>

---

# candle-video

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.82%2B-orange)](https://www.rust-lang.org/)

Библиотека на Rust для генерации видео с использованием AI-моделей, построенная на базе фреймворка [Candle](https://github.com/huggingface/candle). Высокопроизводительный инференс без зависимости от Python.

---

## 📚 Содержание

- [Что это?](#-что-это)
- [Ключевые возможности](#-ключевые-возможности)
- [Демонстрация](#-демонстрация)
- [Системные требования](#-системные-требования)
- [Установка и настройка](#-установка-и-настройка)
- [Как начать использовать](#-как-начать-использовать)
- [Параметры командной строки](#параметры-командной-строки)
- [Поддерживаемые версии моделей](#поддерживаемые-версии-моделей)
- [Оптимизация памяти](#оптимизация-памяти)
- [Структура проекта](#структура-проекта)
- [Благодарности](#-благодарности)
- [Лицензия](#лицензия)

---

## ✨ Что это?

**candle-video** — нативная реализация моделей генерации видео на Rust, ориентированная на сценарии развёртывания, где важны время запуска, размер бинарника и эффективность использования памяти. Обеспечивает инференс современных text-to-video моделей без необходимости Python.

### Поддерживаемые модели

- **[LTX-Video](https://huggingface.co/Lightricks/LTX-Video)** — Генерация видео из текста с архитектурой DiT (Diffusion Transformer)
  - Варианты с 2B и 13B параметрами
  - Стандартные и дистиллированные версии (0.9.5 – 0.9.8)
  - Текстовый энкодер T5-XXL с поддержкой GGUF квантизации
  - 3D VAE для кодирования/декодирования видео
  - Flow Matching планировщик

---

### Роадмап

- [x] **LTX Video 0.9.5-0.9.8**  
- [ ] **Wan 2.1**
- [ ] **Wan 2.2** 
- [ ] **Kandinsky 5**  
- [ ] **LTX-2 (Video only)** 
- [ ] **HunyuanVideo** 
- [ ] **CogVideoX** 
- [ ] **Mochi** 
- [ ] **LTX-2 (Full)**
- [ ] **SVD/SVD XT** 

---

## 🚀 Ключевые возможности

- **Высокая производительность** — Нативный Rust с GPU-ускорением через CUDA/cuDNN
- **Эффективное использование памяти** — BF16 инференс, тайлинг/слайсинг VAE, квантизированные GGUF энкодеры
- **Гибкость** - Работа на CPU или GPU, опциональный Flash Attention v2
- **Автономность** - Не требует Python в продакшене
- **Быстрый запуск** - ~2 секунды против ~15-30 секунд для Python/PyTorch

### Интерфейсы в стиле diffusers

Общие контракты для переиспользования между моделями:
- `DiffusionPipeline` + `PipelineInference` (encode/check/prepare_latents)
- `SchedulerMixin` + `AutoencoderMixin` (операции scheduler, tiling/slicing)
- хуки `Attention` processor для attention модулей трансформеров
- `VideoProcessor` (`preprocess_video` / `postprocess_video`) helpers

См. `src/interfaces` для внутренних трейтов, используемых LTX + SVD.

### Аппаратное ускорение

| Функция | Описание |
|---------|----------|
| `flash-attn` | Flash Attention v2 для эффективного внимания (по умолчанию) |
| `cudnn` | cuDNN для быстрых свёрток (по умолчанию) |
| `mkl` | Intel MKL для оптимизированных CPU операций (x86_64) |
| `accelerate` | Apple Accelerate для Metal (macOS) |
| `nccl` | Мульти-GPU поддержка через NCCL |

---

## 🎬 Демонстрация

| Модель | Видео | Промпт |
| :--- | :---: | :--- |
| **LTX-Video-0.9.5** | ![Waves and Rocks](https://raw.githubusercontent.com/FerrisMind/candle-video/main/examples/ltx-video/output/0.9.5/Waves_and_Rocks.gif) | *The waves crash against the jagged rocks of the shoreline, sending spray high into the air...* |
| **LTX-Video-0.9.8-2b-distilled** | ![woman_with_blood](https://raw.githubusercontent.com/FerrisMind/candle-video/main/examples/ltx-video/output/0.9.8/woman_with_blood.gif) | *A woman with blood on her face and a white tank top looks down and to her right...* |

Больше примеров в [examples](examples/).

---

## 🖥️ Системные требования

### Необходимое ПО

- [**Rust**](https://rust-lang.org/learn/get-started/) 1.82+ (Edition 2024)
- [**CUDA Toolkit**](https://developer.nvidia.com/cuda-12-6-0-download-archive) 12.x (для GPU ускорения)
- [**cuDNN**](https://developer.nvidia.com/cudnn) 8.x/9.x (опционально, для быстрых свёрток)
- [**hf**](https://huggingface.co/docs/huggingface_hub/guides/cli)

### Примерные требования VRAM (512×768, 97 кадров)

- Полная модель: ~8-12GB
- С VAE тайлингом: ~8GB
- С GGUF T5: экономия ~8GB дополнительно

---

## 🛠️ Установка и настройка

### Добавление в проект

```toml
[dependencies]
candle-video = { git = "https://github.com/FerrisMind/candle-video" }
```

### Сборка из исходников

```bash
# Клонирование репозитория
git clone https://github.com/FerrisMind/candle-video.git
cd candle-video

# Сборка по умолчанию (CUDA + cuDNN + Flash Attention)
cargo build --release

# Только CPU
cargo build --release --no-default-features

# С выбранными фичами
cargo build --release --features "cudnn,flash-attn"
```

### Веса моделей

Скачать с [oxide-lab/LTX-Video-0.9.8-2B-distilled](https://huggingface.co/oxide-lab/LTX-Video-0.9.8-2B-distilled):

```bash
hf download oxide-lab/LTX-Video-0.9.8-2B-distilled --local-dir ./models/ltx-video
```

> Примечание: Это та же официальная версия модели `Lightricks/LTX-Video`, но репозиторий содержит все необходимые файлы сразу. Вам не нужно искать всё по отдельности.

**Необходимые файлы для diffusers версий моделей:**
- `transformer/diffusion_pytorch_model.safetensors` — DiT модель
- `vae/diffusion_pytorch_model.safetensors` — 3D VAE
- `text_encoder_gguf/t5-v1_1-xxl-encoder-Q5_K_M.gguf` — Квантизированный T5
- `text_encoder_gguf/tokenizer.json` — T5 токенизатор

**Необходимые файлы для официальных версий моделей:**
- ltxv-2b-0.9.8-distilled.safetensors — DiT + 3D VAE в одном файле
- `text_encoder_gguf/t5-v1_1-xxl-encoder-Q5_K_M.gguf` — Квантизированный T5
- `text_encoder_gguf/tokenizer.json` — T5 токенизатор

---

## 📖 Как начать использовать

### Примеры использования локальных весов (Рекомендуется)

**Для diffusers версий моделей:**

```bash
cargo run --example ltx-video --release --features flash-attn,cudnn -- \
    --local-weights ./models/ltx-video \
    --ltxv-version 0.9.5 \
    --prompt "A cat playing with a ball of yarn" 
```

**Для официальных версий моделей:**

```bash
cargo run --example ltx-video --release --features flash-attn,cudnn -- \
    --local-weights ./models/ltx-video-model \
    --unified-weights ./models/ltx-video-model.safetensors \
    --ltxv-version 0.9.8-2b-distilled \
    --prompt "A cat playing with a ball of yarn" 
```

### Быстрый предпросмотр (Низкое разрешение)

```bash
cargo run --example ltx-video --release --features flash-attn,cudnn -- \
    --local-weights ./models/ltx-video-model \
    --unified-weights ./models/ltx-video-model.safetensors \
    --ltxv-version 0.9.8-2b-distilled \
    --prompt "A cat playing with a ball of yarn" \
    --height 256 --width 384 --num-frames 25 
```

### Режим экономии памяти

```bash
cargo run --example ltx-video --release --features flash-attn,cudnn -- \
    --local-weights ./models/ltx-video \
    --prompt "A majestic eagle soaring over mountains" \
    --vae-tiling --vae-slicing
```

---

## Параметры командной строки

| Аргумент | По умолчанию | Описание |
|----------|--------------|----------|
| `--prompt` | "A video of a cute cat..." | Текстовый промпт |
| `--negative-prompt` | "" | Негативный промпт |
| `--height` | 512 | Высота видео (кратна 32) |
| `--width` | 768 | Ширина видео (кратна 32) |
| `--num-frames` | 97 | Количество кадров (формат 8n + 1) |
| `--steps` | (из конфига версии) | Шаги диффузии |
| `--guidance-scale` | (из конфига версии) | Масштаб classifier-free guidance |
| `--ltxv-version` | "0.9.5" | Версия модели |
| `--local-weights` | (Нет) | Путь к локальным весам |
| `--output-dir` | "output" | Директория для сохранения |
| `--seed` | случайный | Сид для воспроизводимости |
| `--vae-tiling` | false | Включить тайлинг VAE |
| `--vae-slicing` | false | Включить слайсинг VAE |
| `--frames` | false | Сохранять PNG кадры |
| `--gif` | false | Сохранять как GIF |
| `--cpu` | false | Запуск на CPU |
| `--use-bf16-t5` | false | Использовать BF16 T5 вместо GGUF |
| `--unified-weights` | (Нет) | Путь к unified safetensors файлу |

---

## Поддерживаемые версии моделей

| Версия | Параметры | Шаги | Guidance | Примечания |
|--------|-----------|------|----------|------------|
| `0.9.5` | 2B | 40 | 3.0 | Стандартная модель |
| `0.9.6-dev` | 2B | 40 | 3.0 | Версия для разработки |
| `0.9.6-distilled` | 2B | 8 | 1.0 | Быстрый инференс |
| `0.9.8-2b-distilled` | 2B | 7 | 1.0 | Последняя дистиллированная |
| `0.9.8-13b-dev` | 13B | 30 | 8.0 | Большая модель |
| `0.9.8-13b-distilled` | 13B | 7 | 1.0 | Большая дистиллированная |

---

## Оптимизация памяти

Для ограниченной VRAM:

```bash
# VAE тайлинг - обработка изображения тайлами
--vae-tiling

# VAE слайсинг - последовательная обработка батчей
--vae-slicing

# Меньшее разрешение
--height 256 --width 384

# Меньше кадров
--num-frames 25
```

---

## Структура проекта

```
candle-video/
├── src/
│   ├── lib.rs                    # Точка входа библиотеки
│   └── models/
│       └── ltx_video/            # Реализация LTX-Video
│           ├── ltx_transformer.rs    # DiT трансформер
│           ├── vae.rs                # 3D VAE
│           ├── text_encoder.rs       # T5 текстовый энкодер
│           ├── quantized_t5_encoder.rs # GGUF T5 энкодер
│           ├── scheduler.rs          # Flow matching планировщик
│           ├── t2v_pipeline.rs       # Text-to-video пайплайн
│           ├── loader.rs             # Загрузка весов
│           └── configs.rs            # Конфиги версий моделей
├── examples/
│   └── ltx-video/                # Основной CLI пример
├── tests/                        # Тесты паритета и юнит-тесты
├── scripts/                      # Python скрипты для генерации референсов
└── benches/                      # Бенчмарки производительности
```

---

## 🙏 Благодарности

- [Candle](https://github.com/huggingface/candle) — Минималистичный ML фреймворк для Rust
- [Lightricks LTX-Video](https://huggingface.co/Lightricks/LTX-Video) — Оригинальная модель LTX-Video
- [diffusers](https://github.com/huggingface/diffusers) — Референсная реализация

---

## Лицензия

Лицензия Apache License, Version 2.0. Подробности в [LICENSE](LICENSE).

Copyright 2025 FerrisMind
