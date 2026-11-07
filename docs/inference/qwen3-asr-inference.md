# ASR Inference Pipeline ([Qwen3-ASR](https://arxiv.org/abs/2601.21337))

## Overview

```
Audio (16kHz) → [Preprocessing] → [Audio Encoding] → [Text Generation] → Transcription
                 ~5% time          ~20% time           ~75% time
```

## Stage 1: Preprocessing (AudioPreprocessing.swift)

Converts raw audio to a mel spectrogram `[128, T]` using Accelerate framework.

- STFT via `vDSP_fft_zrip` (in-place real FFT, zero-padded 400→512)
- Mel filterbank via `vDSP_mmul`, bin frequencies use padded FFT size (`k * fs / 512`)
- Log-mel via `vDSP_vclip` + `vvlog10f` (vForce vectorized)
- Hann window and FFT setup precomputed once in `init()`
- All temporary buffers preallocated outside the frame loop

## Stage 2: Audio Encoding (AudioEncoder.swift)

18-layer transformer with block attention over chunked mel features.

- Self-attention via `MLXFast.scaledDotProductAttention` (Metal kernel)
- Sinusoidal position embeddings cached by sequence length
- Block attention mask via MLXArray broadcast (`blockIds .== blockIds^T`)

## Stage 3: Text Generation (QuantizedTextDecoder.swift)

28-layer quantized Qwen3 decoder with GQA and RoPE.

- RoPE via `MLXFast.RoPE` (fused Metal kernel)
- GQA via `MLXFast.scaledDotProductAttention` (native GQA support, no manual tiling)
- Causal mask: `nil` for autoregressive steps (seqLen=1), broadcast for prefill
- **Prefill** (seqLen > 1): all prompt tokens in one forward pass
- **Decode** (seqLen = 1): SDPA uses optimized T_q=1 Metal kernel

## Performance

| Model | Framework | RTF | 10s audio processed in |
|-------|-----------|-----|------------------------|
| Qwen3-ASR-0.6B (4-bit) | MLX Swift | ~0.06 | ~0.6s |
| Whisper-large-v3 | whisper.cpp (Q5_0) | ~0.10 | ~1.0s |
| Whisper-small | whisper.cpp (Q5_0) | ~0.04 | ~0.4s |

## Streaming / Partial Transcription

Qwen3-ASR operates in batch mode only. The entire audio input is processed in a single forward pass — there is no streaming or partial transcription support. The audio encoder uses block attention over the full mel spectrogram, and the text decoder generates tokens autoregressively conditioned on the complete encoder output.

For real-time transcription use cases where partial results are needed, consider chunking the audio externally and running separate inference passes on each chunk.

## Language Detection

The model automatically detects the spoken language from the audio content. No language hint or locale parameter is required. The text decoder emits a language token at the start of generation, followed by the transcribed text. Supported languages include English, Chinese, Japanese, Korean, and many European languages.

## Model Architecture Reference

See [docs/models/asr-model.md](../models/asr-model.md) for detailed architecture documentation including layer dimensions, weight formats, and quantization details.

### Local and mixed-precision checkpoints

`Qwen3ASRModel.fromPretrained(modelId:)` accepts a local model directory as well as a HuggingFace ID. Local directories must contain `config.json`, safetensors weights and tokenizer files. The `quantization` block (or `quantization_config` alias) takes precedence over precision inferred from the ID. Affine recipes support global `bits`/`group_size` defaults, checkpoint module path overrides, and `false` to retain floating point. Encoder linears are quantized when explicitly configured or when stored scales identify a quantized encoder; legacy decoder-only recipes retain their float encoder. The tied embedding also has independent precision.

ASR loading verifies every module parameter name and shape, including quantization scales and biases, and rejects unknown tensor namespaces. Both `audio_tower.*`/`model.*` keys and their `thinker.`-prefixed equivalents are accepted. Malformed recipes and incompatible checkpoints throw instead of silently producing invalid transcriptions. Existing public decoder model types and default constructors remain available.

The opt-in `E2EMixedPrecisionParityTests` runner accepts `ASR_PARITY_MODEL`, `ASR_PARITY_INPUT` (JSONL with `wav`, or newline-separated WAV paths), and `ASR_PARITY_OUTPUT`; it writes greedy transcriptions with automatic language detection and load/RSS measurements. `ASR_PARITY_LANGUAGE` optionally forces a language. It skips when the required variables are absent. Checkpoints with per-module recipes use the native mlx-audio automatic-language prompt; decoder-only global recipes retain historical aufklarer prompting.
