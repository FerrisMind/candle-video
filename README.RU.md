# candle-video

[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](https://github.com/FerrisMind/candle-video/blob/main/LICENSE)

**candle-video** — библиотека на Rust для генерации видео с использованием AI-моделей, построенная на базе фреймворка [Candle](https://github.com/huggingface/candle). Обеспечивает высокопроизводительный инференс современных моделей генерации видео.

## Демонстрация

| Модель | Видео | Промпт |
| :--- | :---: | :--- |
| **LTX-Video-0.9.5** | ![Waves and Rocks](https://raw.githubusercontent.com/FerrisMind/candle-video/main/examples/ltx-video/output/0.9.5/Waves_and_Rocks.gif) | *The waves crash against the jagged rocks of the shoreline, sending spray high into the air. The rocks are a dark gray color, with sharp edges and deep crevices. The water is a clear blue-green, with white foam where the waves break against the rocks. The sky is a light gray, with a few white clouds dotting the horizon.* |
| **LTX-Video-0.9.8-2b-distilled** | ![woman_with_blood](https://raw.githubusercontent.com/FerrisMind/candle-video/main/examples/ltx-video/output/0.9.8/woman_with_blood.gif) | *A woman with blood on her face and a white tank top looks down and to her right, then back up as she speaks. She has dark hair pulled back, light skin, and her face and chest are covered in blood. The camera angle is a close-up, focused on the woman's face and upper torso. The lighting is dim and blue-toned, creating a somber and intense atmosphere. The scene appears to be from a movie or TV show.* |
| **Stable Diffusion Video** | *в процессе реализации...* | *в процессе реализации...* |
| **Wan2.1/2.2** | *в планах...* | *в планах...* |

Больше результатов генерации можно найти в[examples](examples/)

🌐 **[English version](README.md)**

## Поддерживаемые модели

- **[LTX-Video](https://huggingface.co/Lightricks/LTX-Video)** — Генерация видео из текста с архитектурой DiT (Diffusion Transformer)
  - Диффузионная модель на базе трансформера
  - Текстовый энкодер T5-XXL (с поддержкой GGUF квантизации)
  - 3D VAE для кодирования/декодирования видео
  - Flow Matching планировщик

## Возможности

- 🚀 **Высокая производительность** — Нативный Rust с GPU-ускорением через CUDA/cuDNN
- 💾 **Эффективное использование памяти** — BF16 инференс, тайлинг/слайсинг VAE, квантизированные GGUF энкодеры
- 🔧 **Гибкость** — Работа на CPU или GPU, опциональный Flash Attention
- 📦 **Автономность** — Не требует Python в продакшене

### Аппаратное ускорение

| Функция | Описание |
|---------|----------|
| `cuda` | CUDA бэкенд для GPU NVIDIA |
| `cudnn` | cuDNN для быстрых свёрток |
| `flash-attn` | Flash Attention v2 для эффективного внимания |
| `mkl` | Intel MKL для оптимизированных CPU операций (x86_64) |
| `accelerate` | Apple Accelerate для Metal (macOS) |
| `nccl` | Мульти-GPU поддержка через NCCL |

## Установка

### Требования

- Rust 1.82+ (edition 2024)
- CUDA Toolkit 12.x (для GPU ускорения)
- cuDNN 8.x/9.x (опционально, для быстрых свёрток)

### Добавление в проект

```toml
[dependencies]
candle-video = { git = "https://github.com/FerrisMind/candle-video" }
```

### Сборка с поддержкой GPU

```bash
# Сборка по умолчанию (CUDA + cuDNN + Flash Attention)
cargo build --release

# Только CPU
cargo build --release --no-default-features

# С выбранными фичами
cargo build --release --features "cudnn,flash-attn"
```

## Быстрый старт

### LTX-Video: Генерация видео из текста

#### Демо

<img src="https://raw.githubusercontent.com/FerrisMind/candle-video/main/examples/ltx-video/output/Demo1.gif" width=50% alt="Demo1" /><br>

#### 1. Автоматический запуск (Рекомендуется)
Веса будут автоматически скачаны из [oxide-lab/LTX-Video-0.9.5](https://huggingface.co/oxide-lab/LTX-Video-0.9.5).

```bash
cargo run --example ltx-video --release --features flash-attn,cudnn -- \
    --prompt "A futuristic cityscape with flying cars"
```

#### 2. Ручной запуск (Локальные веса)
Если у вас уже есть веса, укажите путь к ним:

```bash
cargo run --example ltx-video --release --features flash-attn,cudnn -- \
    --local-weights ./models/ltx-video \
    --prompt "A futuristic cityscape with flying cars" \
    --vae-tiling
```
# Режим экономии памяти (с тайлингом VAE)
```bash
cargo run --example ltx-video --release --features flash-attn,cudnn -- \
    --local-weights ./models/ltx-video \
    --prompt "A futuristic cityscape with flying cars" \
    --vae-tiling --vae-slicing
```

Подробности в [examples/ltx-video](examples/ltx-video/README.md).

### Параметры командной строки

| Аргумент | По умолчанию | Описание |
|----------|--------------|----------|
| `--prompt` | "A video of a cute cat..." | Текстовый промпт |
| `--negative-prompt` | "" | Негативный промпт |
| `--height` | 512 | Высота (должна быть кратна 32) |
| `--width` | 768 | Ширина (должна быть кратна 32) |
| `--num-frames` | 97 | Количество кадров (формат 8n + 1) |
| `--steps` | (из конфига версии) | Шаги диффузии (40 для 0.9.5, 8 для distilled) |
| `--guidance-scale` | (из конфига версии) | Масштаб classifier-free guidance |
| `--ltxv-version` | "0.9.5" | Версия модели (0.9.5, 0.9.6-distilled, 0.9.8-2b-distilled и др.) |
| `--local-weights` | (Нет) | Путь к локальным весам (авто-скачивание если не указан) |
| `--output-dir` | "output" | Директория для сохранения результатов |
| `--seed` | случайный | Сид для воспроизводимости |
| `--vae-tiling` | false | Включить тайлинг VAE для экономии памяти |
| `--vae-slicing` | false | Включить слайсинг VAE для батчей |
| `--frames` | false | Сохранять PNG кадры (отключает GIF) |
| `--cpu` | false | Запуск на CPU вместо GPU |
| `--model-id` | "Lightricks/LTX-Video" | ID модели HF (для скачивания токенизатора) |
| `--use-bf16-t5` | false | Использовать BF16 T5 вместо GGUF квантизированного |
| `--unified-weights` | (Нет) | Путь к unified safetensors файлу (официальный формат LTX) |

### Использование как библиотеки

```rust
use candle_core::{Device, DType};
use candle_video::models::ltx_video::{
    LtxVideoTransformer3DModel,
    AutoencoderKLLtxVideo,
    FlowMatchEulerDiscreteScheduler,
    loader::WeightLoader,
};

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let dtype = DType::BF16;
    
    // Загрузка Transformer
    let loader = WeightLoader::new(device.clone(), dtype);
    let vb = loader.load_single("path/to/transformer.safetensors")?;
    let config = LtxVideoTransformer3DModelConfig::default();
    let transformer = LtxVideoTransformer3DModel::new(&config, vb)?;
    
    // Загрузка VAE
    let vae_vb = loader.load_single("path/to/vae.safetensors")?;
    let mut vae = AutoencoderKLLtxVideo::new(
        AutoencoderKLLtxVideoConfig::default(),
        vae_vb
    )?;
    
    // Включение оптимизаций памяти
    vae.use_tiling = true;
    vae.use_slicing = true;
    
    // ... настройка пайплайна и генерация
    Ok(())
}
```

## Структура проекта

```
candle-video/
├── src/
│   ├── lib.rs              # Точка входа библиотеки
│   ├── models/
│   │   ├── ltx_video/      # Компоненты модели LTX-Video
│   │   │   ├── ltx_transformer.rs    # DiT трансформер
│   │   │   ├── vae.rs                # 3D VAE
│   │   │   ├── text_encoder.rs       # T5 текстовый энкодер
│   │   │   ├── quantized_t5_encoder.rs # GGUF T5 энкодер
│   │   │   ├── scheduler.rs          # Flow matching планировщик
│   │   │   ├── t2v_pipeline.rs       # Text-to-video пайплайн
│   │   │   └── loader.rs             # Утилиты загрузки весов
│   │   └── svd/            # Компоненты Stable Video Diffusion
│   │       ├── unet/       # UNet архитектура
│   │       ├── vae/        # Временной VAE
│   │       ├── clip.rs     # CLIP энкодер изображений
│   │       ├── pipeline.rs # Пайплайн генерации
│   │       └── scheduler.rs# EulerA планировщик
│   └── utils/              # Утилиты
├── examples/               # Примеры использования (запуск через --example)
│   ├── ltx_video/          # Пример генерации видео
│   │   ├── main.rs         # Точка входа
│   │   └── README.md       # Подробное описание
│   └── verify/             # Инструменты верификации и отладки
├── scripts/                # Python скрипты для верификации
├── tests/                  # Интеграционные тесты
├── prebuilt/               # Прекомпилированные ядра
└── tp/                     # Сторонние подмодули
```

## Веса моделей

### LTX-Video

Скачать с [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video):

```bash
# Через huggingface-cli
huggingface-cli download Lightricks/LTX-Video --local-dir ./models/ltx-video

# Для GGUF T5 энкодера (экономия памяти)
# Скачать t5-v1_1-xxl-encoder-Q5_K_M.gguf
```

**Необходимые файлы весов:**
- `transformer/diffusion_pytorch_model.safetensors` — DiT модель
- `vae/diffusion_pytorch_model.safetensors` — 3D VAE
- `text_encoder_gguf/t5-v1_1-xxl-encoder-Q5_K_M.gguf` — Квантизированный T5
- `tokenizer/tokenizer.json` — T5 токенизатор

## Оптимизация памяти

Для ограниченной VRAM включите эти опции:

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

**Примерные требования VRAM (512x768, 97 кадров):**
- Полная модель: ~8-12GB
- C VAE тайлингом: ~8GB
- С GGUF T5: экономия ~8GB

## Сравнение с PyTorch/diffusers

| Характеристика | candle-video | diffusers (Python) |
|----------------|-------------|-------------------|
| Рантайм | Нативный Rust | Python + PyTorch |
| Запуск | ~2 секунды | ~15-30 секунд |
| Размер бинарника | ~50MB | 2GB+ (с зависимостями) |
| Использование VRAM | Оптимизировано | Стандартное |
| Деплой | Один бинарник | Python окружение |

## Частые проблемы

### CUDA не найдена

```bash
# Убедитесь, что CUDA в PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Ошибки cuDNN на Windows

Скопируйте и переименуйте эти DLL в PATH:
- `nvcuda.dll` → `cuda.dll`
- `cublas64_12.dll` → `cublas.dll`
- `curand64_10.dll` → `curand.dll`

### Нехватка памяти

Уменьшите разрешение, количество кадров или включите VAE тайлинг:
```bash
--height 256 --width 384 --num-frames 25 --vae-tiling
```

## Вклад в проект

Вклады приветствуются! Открывайте issue или pull request.

## Лицензия

Лицензия Apache License, Version 2.0. Подробности в [LICENSE](LICENSE).

## Благодарности

- [Candle](https://github.com/huggingface/candle) — Минималистичный ML фреймворк для Rust
- [Lightricks LTX-Video](https://huggingface.co/Lightricks/LTX-Video) — Оригинальная модель LTX-Video
- [Stability AI](https://stability.ai/) — Stable Video Diffusion
- [diffusers](https://github.com/huggingface/diffusers) — Референсная реализация
