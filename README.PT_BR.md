</p>
<p align="left">
  <a href="README.md"><img src="https://img.shields.io/badge/English-232323" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-232323" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-5B7CFA" alt="Português"></a>
</p>

---

# candle-video

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.82%2B-orange)](https://www.rust-lang.org/)

Biblioteca Rust para geração de vídeo com IA, construída sobre o framework [Candle](https://github.com/huggingface/candle). Inferência de alto desempenho sem dependência do Python.

> **Tradução:** Talita Maia Sousa

---

## 📚 Índice

- [O que é isso?](#-o-que-é-isso)
- [Principais Recursos](#-principais-recursos)
- [Demonstração](#-demonstração)
- [Requisitos do Sistema](#-requisitos-do-sistema)
- [Instalação e Configuração](#-instalação-e-configuração)
- [Como Começar a Usar](#-como-começar-a-usar)
- [Opções de Linha de Comando](#opções-de-linha-de-comando)
- [Versões de Modelos Suportadas](#versões-de-modelos-suportadas)
- [Otimização de Memória](#otimização-de-memória)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Agradecimentos](#-agradecimentos)
- [Licença](#licença)

---

## ✨ O que é isso?

**candle-video** é uma implementação nativa em Rust de modelos de geração de vídeo, voltada para cenários de implantação onde tempo de inicialização, tamanho do binário e eficiência de memória são importantes. Fornece inferência para modelos text-to-video de última geração sem necessidade de runtime Python.

### Modelos Suportados

- **[LTX-Video](https://huggingface.co/Lightricks/LTX-Video)** — Geração de vídeo a partir de texto usando arquitetura DiT (Diffusion Transformer)
  - Variantes com 2B e 13B parâmetros
  - Versões padrão e destiladas (0.9.5 – 0.9.8)
  - Codificador de texto T5-XXL com suporte a quantização GGUF
  - VAE 3D para codificação/decodificação de vídeo
  - Scheduler Flow Matching

### Roteiro

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

## 🚀 Principais Recursos

- **Alto Desempenho** — Rust nativo com aceleração GPU via CUDA/cuDNN
- **Eficiência de Memória** — Inferência BF16, tiling/slicing de VAE, codificadores GGUF quantizados
- **Flexível** — Execute em CPU ou GPU, com Flash Attention v2 opcional
- **Autônomo** — Não requer runtime Python em produção
- **Inicialização Rápida** — ~2 segundos vs ~15-30 segundos para Python/PyTorch

### Aceleração de Hardware

| Recurso | Descrição |
|---------|-----------|
| `flash-attn` | Flash Attention v2 para atenção eficiente (padrão) |
| `cudnn` | cuDNN para convoluções mais rápidas (padrão) |
| `mkl` | Intel MKL para operações CPU otimizadas (x86_64) |
| `accelerate` | Apple Accelerate para Metal (macOS) |
| `nccl` | Suporte multi-GPU via NCCL |

---

## 🎬 Demonstração

| Modelo | Vídeo | Prompt |
| :--- | :---: | :--- |
| **LTX-Video-0.9.5** | ![Waves and Rocks](https://raw.githubusercontent.com/FerrisMind/candle-video/main/examples/ltx-video/output/0.9.5/Waves_and_Rocks.gif) | *The waves crash against the jagged rocks of the shoreline, sending spray high into the air...* |
| **LTX-Video-0.9.8-2b-distilled** | ![woman_with_blood](https://raw.githubusercontent.com/FerrisMind/candle-video/main/examples/ltx-video/output/0.9.8/woman_with_blood.gif) | *A woman with blood on her face and a white tank top looks down and to her right...* |

Mais exemplos em [examples](examples/).

---

## 🖥️ Requisitos do Sistema

### Pré-requisitos

- [**Rust**](https://rust-lang.org/learn/get-started/) 1.82+ (Edition 2024)
- [**CUDA Toolkit**](https://developer.nvidia.com/cuda-12-6-0-download-archive) 12.x (para aceleração GPU)
- [**cuDNN**](https://developer.nvidia.com/cudnn) 8.x/9.x (opcional, para convoluções mais rápidas)
- [**hf**](https://huggingface.co/docs/huggingface_hub/guides/cli)

### Requisitos Aproximados de VRAM (512×768, 97 frames)

- Modelo completo: ~8-12GB
- Com VAE tiling: ~8GB
- Com GGUF T5: economia de ~8GB adicionais

---

## 🛠️ Instalação e Configuração

### Adicionar ao seu projeto

```toml
[dependencies]
candle-video = { git = "https://github.com/FerrisMind/candle-video" }
```

### Compilar a partir do código fonte

```bash
# Clonar o repositório
git clone https://github.com/FerrisMind/candle-video.git
cd candle-video

# Compilação padrão (CUDA + cuDNN + Flash Attention)
cargo build --release

# Compilação apenas CPU
cargo build --release --no-default-features

# Com recursos específicos
cargo build --release --features "cudnn,flash-attn"
```

### Pesos dos Modelos

Baixar de [oxide-lab/LTX-Video-0.9.8-2B-distilled](https://huggingface.co/oxide-lab/LTX-Video-0.9.8-2B-distilled):

```bash
hf download oxide-lab/LTX-Video-0.9.8-2B-distilled --local-dir ./models/ltx-video
```

> Nota: Esta é a mesma versão oficial do modelo `Lightricks/LTX-Video`, mas o repositório contém todos os arquivos necessários de uma vez. Você não precisa procurar tudo individualmente.

**Arquivos necessários para versões diffusers dos modelos:**
- `transformer/diffusion_pytorch_model.safetensors` — Modelo DiT
- `vae/diffusion_pytorch_model.safetensors` — VAE 3D
- `text_encoder_gguf/t5-v1_1-xxl-encoder-Q5_K_M.gguf` — T5 quantizado
- `text_encoder_gguf/tokenizer.json` — Tokenizador T5

**Arquivos necessários para versões oficiais dos modelos:**
- ltxv-2b-0.9.8-distilled.safetensors — DiT + VAE 3D em arquivo único
- `text_encoder_gguf/t5-v1_1-xxl-encoder-Q5_K_M.gguf` — T5 quantizado
- `text_encoder_gguf/tokenizer.json` — Tokenizador T5

---

## 📖 Como Começar a Usar

### Exemplos Usando Pesos Locais (Recomendado)

**Para versões diffusers dos modelos:**

```bash
cargo run --example ltx-video --release --features flash-attn,cudnn -- \
    --local-weights ./models/ltx-video \
    --ltxv-version 0.9.5 \
    --prompt "A cat playing with a ball of yarn" 
```

**Para versões oficiais dos modelos:**

```bash
cargo run --example ltx-video --release --features flash-attn,cudnn -- \
    --local-weights ./models/ltx-video-model \
    --unified-weights ./models/ltx-video-model.safetensors \
    --ltxv-version 0.9.8-2b-distilled \
    --prompt "A cat playing with a ball of yarn" 
```

### Pré-visualização Rápida (Resolução Menor)

```bash
cargo run --example ltx-video --release --features flash-attn,cudnn -- \
    --local-weights ./models/ltx-video-model \
    --unified-weights ./models/ltx-video-model.safetensors \
    --ltxv-version 0.9.8-2b-distilled \
    --prompt "A cat playing with a ball of yarn" \
    --height 256 --width 384 --num-frames 25 
```

### Modo de Baixa VRAM

```bash
cargo run --example ltx-video --release --features flash-attn,cudnn -- \
    --local-weights ./models/ltx-video \
    --prompt "A majestic eagle soaring over mountains" \
    --vae-tiling --vae-slicing
```

---

## Opções de Linha de Comando

| Argumento | Padrão | Descrição |
|-----------|--------|-----------|
| `--prompt` | "A video of a cute cat..." | Prompt de texto para geração |
| `--negative-prompt` | "" | Prompt negativo |
| `--height` | 512 | Altura do vídeo (divisível por 32) |
| `--width` | 768 | Largura do vídeo (divisível por 32) |
| `--num-frames` | 97 | Número de frames (deve ser 8n + 1) |
| `--steps` | (do config da versão) | Passos de difusão |
| `--guidance-scale` | (do config da versão) | Escala de classifier-free guidance |
| `--ltxv-version` | "0.9.5" | Versão do modelo |
| `--local-weights` | (Nenhum) | Caminho para pesos locais |
| `--output-dir` | "output" | Diretório para salvar resultados |
| `--seed` | aleatório | Seed para reprodutibilidade |
| `--vae-tiling` | false | Habilitar tiling de VAE |
| `--vae-slicing` | false | Habilitar slicing de VAE |
| `--frames` | false | Salvar frames PNG individuais |
| `--gif` | false | Salvar como animação GIF |
| `--cpu` | false | Executar em CPU |
| `--use-bf16-t5` | false | Usar T5 BF16 em vez de GGUF quantizado |
| `--unified-weights` | (Nenhum) | Caminho para arquivo safetensors unificado |

---

## Versões de Modelos Suportadas

| Versão | Parâmetros | Passos | Guidance | Notas |
|--------|------------|--------|----------|-------|
| `0.9.5` | 2B | 40 | 3.0 | Modelo padrão |
| `0.9.6-dev` | 2B | 40 | 3.0 | Versão de desenvolvimento |
| `0.9.6-distilled` | 2B | 8 | 1.0 | Inferência rápida |
| `0.9.8-2b-distilled` | 2B | 7 | 1.0 | Última destilada |
| `0.9.8-13b-dev` | 13B | 30 | 8.0 | Modelo grande |
| `0.9.8-13b-distilled` | 13B | 7 | 1.0 | Grande destilada |

---

## Otimização de Memória

Para VRAM limitada:

```bash
# VAE tiling - processa imagem em tiles
--vae-tiling

# VAE slicing - processa batches sequencialmente
--vae-slicing

# Resolução menor
--height 256 --width 384

# Menos frames
--num-frames 25
```

---

## Estrutura do Projeto

```
candle-video/
├── src/
│   ├── lib.rs                    # Ponto de entrada da biblioteca
│   └── models/
│       └── ltx_video/            # Implementação LTX-Video
│           ├── ltx_transformer.rs    # Transformer DiT
│           ├── vae.rs                # VAE 3D
│           ├── text_encoder.rs       # Codificador de texto T5
│           ├── quantized_t5_encoder.rs # Codificador T5 GGUF
│           ├── scheduler.rs          # Scheduler Flow matching
│           ├── t2v_pipeline.rs       # Pipeline text-to-video
│           ├── loader.rs             # Carregamento de pesos
│           └── configs.rs            # Configs de versões de modelos
├── examples/
│   └── ltx-video/                # Exemplo CLI principal
├── tests/                        # Testes de paridade e unitários
├── scripts/                      # Scripts Python para geração de referências
└── benches/                      # Benchmarks de desempenho
```

---

## 🙏 Agradecimentos

- [Candle](https://github.com/huggingface/candle) — Framework ML minimalista para Rust
- [Lightricks LTX-Video](https://huggingface.co/Lightricks/LTX-Video) — Modelo LTX-Video original
- [diffusers](https://github.com/huggingface/diffusers) — Implementação de referência

---

## Licença

Licenciado sob a Apache License, Version 2.0. Veja [LICENSE](LICENSE) para detalhes.

Copyright 2025 FerrisMind
