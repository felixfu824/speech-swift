# speech-swift (Lamitype fork)

This is a fork of [soniqo/speech-swift](https://github.com/soniqo/speech-swift). It changes the Qwen3-ASR path so that [Lamitype](https://github.com/felixfu824/Lamitype) can run a mixed 6/8-bit Qwen3-ASR model on device. For everything else (other models, installation, full API docs), see the original repository.

This branch is based on upstream commit `d3f159c` (2026-03-31). It is not rebased onto current upstream.

## What this fork changes

1. **Cache-first model loading** (`d603472`). A fully cached model loads without contacting the Hugging Face Hub, so loading works offline and during Hub outages. The network is used only for a first download or an incomplete cache.
2. **Mixed-precision checkpoints** (`bc699d6`). Qwen3-ASR loads checkpoints whose `quantization` config sets bits and group size per module, including a quantized audio encoder and embedding. Loading checks every parameter name and shape and rejects incompatible checkpoints. A local model folder also works in place of a Hugging Face ID.
3. **Forced ASR prompt for every checkpoint** (`9399f00`). All checkpoints use the `<asr_text>` prompt that the 4-bit models already used. With the model's own auto-language prompt, short mixed Chinese and English clips were sometimes written in Thai, Arabic or Korean script. Set `usesAutoLanguagePrompt = true` on the model to get the old behaviour.

Details: [docs/inference/qwen3-asr-inference.md](docs/inference/qwen3-asr-inference.md).

## Use

```swift
.package(url: "https://github.com/felixfu824/speech-swift", branch: "main")
```

To rebuild a Lamitype release exactly, use the speech-swift commit pinned in that release's `Package.swift`.

```swift
import Qwen3ASR

let model = try await Qwen3ASRModel.fromPretrained(modelId: "felixfu824/Qwen3-ASR-0.6B-MLX-mixed-6-8bit")
let text = model.transcribe(audio: samples, sampleRate: 16000)
```

Model: [felixfu824/Qwen3-ASR-0.6B-MLX-mixed-6-8bit](https://huggingface.co/felixfu824/Qwen3-ASR-0.6B-MLX-mixed-6-8bit).

## License

Apache License 2.0, the same as upstream. See [LICENSE](LICENSE). The changes in this fork are listed above.
