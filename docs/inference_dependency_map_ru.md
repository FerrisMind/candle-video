# Карта Зависимостей для Вывода (Inference Dependency Map)

Этот документ содержит минимальный набор файлов, необходимых для вывода (inference) моделей **LTX Video** и **Stable Video Diffusion (SVD)** в библиотеке `diffusers`.

## 1. Модель LTX Video
**Основные Файлы Вывода:**
*   `src/diffusers/pipelines/ltx/pipeline_ltx.py` (Пайплайн Text-to-Video)
*   `src/diffusers/pipelines/ltx/pipeline_ltx_image2video.py` (Пайплайн Image-to-Video)
*   `src/diffusers/pipelines/ltx/pipeline_output.py` (Выходной формат пайплайна `LTXPipelineOutput`)
*   `src/diffusers/pipelines/ltx/__init__.py` (Инициализация модуля)
*   `src/diffusers/models/transformers/transformer_ltx.py` (Трансформер / Backbone, включает `LTXVideoTransformerBlock`, `LTXVideoAttnProcessor`)
*   `src/diffusers/models/autoencoders/autoencoder_kl_ltx.py` (VAE)
*   `src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py` (Планировщик / Scheduler)

**Специфические Зависимости Компонентов:**
*   `src/diffusers/models/attention_dispatch.py` (Диспетчер функций внимания)
*   `src/diffusers/video_processor.py` (Используется пайплайном)

**Дополнительные/Опциональные Файлы (Для полной совместимости):**
*   `src/diffusers/pipelines/ltx/pipeline_ltx_condition.py` (Пайплайн с дополнительными условиями)
*   `src/diffusers/pipelines/ltx/pipeline_ltx_latent_upsample.py` (Пайплайн для апскейлинга латентов)
*   `src/diffusers/pipelines/ltx/modeling_latent_upsampler.py` (Модель апскейлера)

## 2. Модель Stable Video Diffusion (SVD)
**Основные Файлы Вывода:**
*   `src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py` (Пайплайн вывода)
*   `src/diffusers/pipelines/stable_video_diffusion/__init__.py` (Инициализация модуля)
*   `src/diffusers/models/unets/unet_spatio_temporal_condition.py` (UNet Backbone)
*   `src/diffusers/models/autoencoders/autoencoder_kl_temporal_decoder.py` (VAE с временным декодером)
*   `src/diffusers/schedulers/scheduling_euler_discrete.py` (Планировщик / Scheduler)

**Специфические Зависимости Компонентов:**
*   `src/diffusers/models/unets/unet_3d_blocks.py` (Активно используется в UNet и VAE)
*   `src/diffusers/models/unets/unet_2d_blocks.py` (Блоки для 2D UNet, используется в VAE)
*   `src/diffusers/models/unets/unet_motion_model.py` (Модули движения)
*   `src/diffusers/models/transformers/transformer_temporal.py` (Временные трансформеры, используемые в 3D блоках)
*   `src/diffusers/models/transformers/transformer_2d.py` (2D Трансформеры, используемые в 3D блоках)
*   `src/diffusers/models/autoencoders/vae.py` (Базовые классы и компоненты VAE: `Encoder`, `Decoder`, `DiagonalGaussianDistribution`)
*   `src/diffusers/video_processor.py` (Используется пайплайном)
*   `src/diffusers/image_processor.py` (Обработка входных изображений)

## 3. Общие Базовые Зависимости (Необходимы для ОБЕИХ моделей)
Эти файлы содержат базовые классы, утилиты и общие слои. Их удаление приведет к неработоспособности вывода.

**Базовые Утилиты и Конфигурация:**
*   `src/diffusers/configuration_utils.py` (`ConfigMixin`, `register_to_config`)
*   `src/diffusers/models/modeling_utils.py` (`ModelMixin`)
*   `src/diffusers/models/modeling_outputs.py` (Обработка выходных данных)
*   `src/diffusers/models/_modeling_parallel.py` (Примитивы Context Parallelism)
*   `src/diffusers/pipelines/pipeline_utils.py` (Базовый класс `DiffusionPipeline`)
*   `src/diffusers/pipelines/pipeline_loading_utils.py` (Логика загрузки)
*   `src/diffusers/schedulers/scheduling_utils.py` (`SchedulerMixin`)
*   `src/diffusers/callbacks.py` (Callback-классы для пайплайнов)
*   `src/diffusers/utils/` (Сильная зависимость от `__init__.py`, `logging.py`, `doc_utils.py`, `import_utils.py`, `outputs.py`, `torch_utils.py`, `accelerate_utils.py`)

**Общие Компоненты Модели:**
*   `src/diffusers/models/attention_processor.py` (Процессоры внимания)
*   `src/diffusers/models/attention.py` (Базовые слои внимания)
*   `src/diffusers/models/attention_dispatch.py` (Диспетчер функций внимания)
*   `src/diffusers/models/embeddings.py` (Временные и позиционные эмбеддинги)
*   `src/diffusers/models/activations.py` (Функции активации, например GELU)
*   `src/diffusers/models/normalization.py` (LayerNorm, RMSNorm, AdaLayerNorm)
*   `src/diffusers/models/resnet.py` (Блоки ResNet, Down/Upsample, `SpatioTemporalResBlock`, `TemporalConvLayer`)
*   `src/diffusers/models/downsampling.py` (Слои понижения разрешения)
*   `src/diffusers/models/upsampling.py` (Слои повышения разрешения)
*   `src/diffusers/models/cache_utils.py` (Механизмы кэширования)
*   `src/diffusers/loaders/__init__.py` (Миксины загрузчиков)
*   `src/diffusers/loaders/single_file.py` (Поддержка загрузки одиночных файлов)
*   `src/diffusers/loaders/lora_base.py` (Базовые миксины LoRA)
*   `src/diffusers/loaders/peft.py` (Интеграция с PEFT)
*   `src/diffusers/loaders/unet.py` (Миксины загрузки UNet, `UNet2DConditionLoadersMixin`)

**Файлы инициализации модулей (`__init__.py`):**
*   `src/diffusers/__init__.py`
*   `src/diffusers/models/__init__.py`
*   `src/diffusers/models/transformers/__init__.py`
*   `src/diffusers/models/autoencoders/__init__.py`
*   `src/diffusers/models/unets/__init__.py`
*   `src/diffusers/pipelines/__init__.py`
*   `src/diffusers/schedulers/__init__.py`

**Зависимости от Внешних Библиотек (Критические):**
*   `torch`
*   `numpy`
*   `transformers` (Для текстовых энкодеров/токенизаторов, `CLIPImageProcessor`, `CLIPVisionModelWithProjection` и т.д.)
*   `huggingface_hub`
*   `safetensors`
*   `accelerate` (Опционально, но глубоко интегрировано)

## Итог по "Минимальному Набору"
Для поддержки *только* этих двух моделей вам необходимо сохранить все специфические файлы моделей, перечисленные в Разделах 1 и 2, ПЛЮС весь набор файлов из Раздела 3.

> **⚠️ Важно:** Директория `src/diffusers/models/` имеет сильные внутренние связи:
> - `unet_3d_blocks.py` импортирует `resnet.py`, `transformer_2d.py`, `transformer_temporal.py`, `unet_motion_model.py`
> - `autoencoder_kl_temporal_decoder.py` зависит от `vae.py` и `unet_3d_blocks.py`
> - `vae.py` зависит от `unet_2d_blocks.py`
