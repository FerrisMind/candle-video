LTX-Video 0.9.8 2B Candle Inference (t2v + i2v)
Прогресс: 100% (29/29 задач)
Этап 1: Формы латентов и конфиги
[x] Исправить формулу latent_t в TextToVideoPipeline::compute_latent_dims()
[x] Синхронизировать вычисление latent dims в библиотеке с формулой diffusers (num_frames - 1) / 8 + 1
[x] Добавить юнит-тест на latent_t по формуле diffusers
Этап 2: RoPE паритет с diffusers
[x] Зафиксировать RoPE-константы в rope.rs (base_num_frames=20, base_height=2048, base_width=2048, theta=10000)
[x] Добавить DEFAULT_FRAME_RATE=25 и frame_rate: Option<f64> в публичный конфиг генерации
[x] Реализовать вычисление rope_interpolation_scale=(8/frame_rate, 32, 32) как в diffusers
[x] Обновить генерацию indices_grid в rope.rs под контракт (B, 3, seq_len) и scaling как в diffusers
[x] Перевести ltx_video.rs на новый indices_grid с diffusers-параметрами
[x] Добавить тест на генерацию indices_grid с ожидаемой формой (B, 3, seq_len)
Этап 3: Реальный t2v в библиотечном pipeline
[x] Добавить API загрузки DiT из ltxv-2b-0.9.8-distilled.safetensors с префиксом model.diffusion_model
[x] Добавить API загрузки VAE decoder из embedded vae.decoder или отдельного vae.safetensors
[x] Добавить загрузку per_channel_statistics для denormalize latents
[x] Интегрировать scheduler.set_timesteps_with_shape() перед денойзингом
[x] Реализовать TextToVideoPipeline::generate() без mock_* (denoise -> denormalize -> VAE decode)
[x] Реализовать TextToVideoPipeline::generate_with_cfg() без mock_* (CFG -> scheduler step -> decode)
Этап 4: Реальный i2v (один кадр, idx=0)
[x] Добавить API загрузки VAE encoder (embedded vae.encoder или vae.safetensors префикс encoder)
[x] Реализовать преобразование входного изображения в тензор [B, 3, T=1, H, W] в диапазоне [-1, 1]
[x] Реализовать VAE encode изображения в conditioning latents
[x] Реализовать conditioning mask для conditioning_frame_index=0
[x] Реализовать i2v денойзинг через compute_token_timesteps + step_per_token + apply_conditioning_mask
[x] Реализовать публичный generate_image_to_video() без mock_*
Этап 5: CLI и регрессионные проверки
[x] Добавить --frame-rate в ltx_video.rs (default 25)
[x] Добавить --mode t2v|i2v и --image в ltx_video.rs
[x] Переключить CLI на вызовы библиотечного TextToVideoPipeline для t2v/i2v
Этап 6: Поддержка F32 и FP8
[x] Добавить флаги --f32 и --fp8 в ltx_video.rs
[x] Обновить выбор dtype с валидацией конфликтующих флагов
Этап 7: Flash-Attn для экономии памяти
[x] Добавить зависимость candle-flash-attn и feature flash-attn
[x] Добавить ветку flash-attn в attention при отсутствии маски
[x] Добавить флаг --use-flash-attn и прокинуть в DitConfig
