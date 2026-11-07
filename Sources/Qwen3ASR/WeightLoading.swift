import Foundation
import MLXCommon
import MLX
import MLXNN
import AudioCommon

/// Weight loading utilities for Qwen3-ASR
/// Uses direct HuggingFace key paths - model structure must match exactly
public enum WeightLoader {

    /// Load weights from safetensors file
    public static func loadSafetensors(url: URL) throws -> [String: MLXArray] {
        try CommonWeightLoader.loadSafetensors(url: url)
    }

    /// Load and apply weights to model using HuggingFace key paths directly
    public static func loadWeights(
        into audioEncoder: Qwen3AudioEncoder,
        from directory: URL
    ) throws {
        let weights = try normalizedWeights(from: directory)
        try applyStrict(audioEncoder, weights: weights, prefix: "audio_tower.")
    }

    public static func loadTextDecoderWeights(
        into textModel: QuantizedTextModel, from directory: URL
    ) throws {
        try applyStrict(textModel, weights: normalizedWeights(from: directory), prefix: "model.")
    }

    static func normalizedWeights(from directory: URL) throws -> [String: MLXArray] {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }.sorted { $0.path < $1.path }
        guard !files.isEmpty else { throw WeightLoadingError.noWeightsFound(directory) }
        var weights: [String: MLXArray] = [:]
        for file in files {
            for (key, value) in try loadSafetensors(url: file) {
                guard weights.updateValue(value, forKey: key) == nil else {
                    throw WeightLoadingError.incompatibleWeights("Duplicate tensor across shards: \(key)")
                }
            }
        }
        var result: [String: MLXArray] = [:]
        for (original, value) in weights {
            let key = original.hasPrefix("thinker.") ? String(original.dropFirst(8)) : original
            guard key.hasPrefix("audio_tower.") || key.hasPrefix("model.") else {
                throw WeightLoadingError.incompatibleWeights("Unexpected tensor: \(original)")
            }
            guard result.updateValue(value, forKey: key) == nil else {
                throw WeightLoadingError.incompatibleWeights("Duplicate tensor: \(key)")
            }
        }
        return result
    }

    static func applyStrict(_ module: Module, weights: [String: MLXArray], prefix: String) throws {
        var selected = Dictionary(uniqueKeysWithValues: weights.compactMap { key, value in
            key.hasPrefix(prefix) ? (String(key.dropFirst(prefix.count)), value) : nil
        })
        if let text = module as? QuantizedTextModel, text.embedTokens.bits == 0 {
            for key in ["scales", "biases"] {
                let path = "embed_tokens." + key
                guard selected[path] == nil else { throw WeightLoadingError.incompatibleWeights("Unexpected float embedding tensor: \(path)") }
                selected[path] = key == "scales" ? text.embedTokens.scales : text.embedTokens.biases
            }
        }
        try module.update(parameters: ModuleParameters.unflattened(selected), verify: .all)
    }

    // MARK: - Forced Aligner Weight Loading

    /// Load weights for the forced aligner model.
    /// Weight key structure:
    ///   - `thinker.audio_tower.*` → audio encoder
    ///   - `thinker.model.*` → text decoder
    ///   - `thinker.lm_head.*` → classify head (Linear, NOT quantized)
    public static func loadForcedAlignerWeights(
        into model: Qwen3ForcedAligner,
        from directory: URL
    ) throws {
        let fileManager = FileManager.default
        let contents = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        let safetensorFiles = contents.filter { $0.pathExtension == "safetensors" }

        guard !safetensorFiles.isEmpty else {
            throw WeightLoadingError.noWeightsFound(directory)
        }

        // Load all weight files
        var allWeights: [String: MLXArray] = [:]
        for file in safetensorFiles {
            print("Loading: \(file.lastPathComponent)")
            let weights = try loadSafetensors(url: file)
            allWeights.merge(weights) { _, new in new }
        }

        print("Loaded \(allWeights.count) weight tensors")

        // Strip thinker. prefix and categorize
        var audioTowerWeights: [String: MLXArray] = [:]
        var textWeights: [String: MLXArray] = [:]
        var classifyWeights: [String: MLXArray] = [:]

        for (key, value) in allWeights {
            // Strip thinker. prefix if present
            let stripped: String
            if key.hasPrefix("thinker.") {
                stripped = String(key.dropFirst("thinker.".count))
            } else {
                stripped = key
            }

            if stripped.hasPrefix("audio_tower.") {
                let audioKey = String(stripped.dropFirst("audio_tower.".count))
                audioTowerWeights[audioKey] = value
            } else if stripped.hasPrefix("model.") {
                let modelKey = String(stripped.dropFirst("model.".count))
                textWeights[modelKey] = value
            } else if stripped.hasPrefix("lm_head.") {
                classifyWeights[stripped] = value
            }
        }

        print("Audio tower: \(audioTowerWeights.count), Text decoder: \(textWeights.count), Classify head: \(classifyWeights.count)")

        // Load audio encoder weights (transpose Conv2d from PyTorch [outC,inC,kH,kW] to MLX [outC,kH,kW,inC])
        applyConv2dWeights(to: model.audioEncoder.conv2d1, prefix: "conv2d1", from: audioTowerWeights, transposePyTorch: true)
        applyConv2dWeights(to: model.audioEncoder.conv2d2, prefix: "conv2d2", from: audioTowerWeights, transposePyTorch: true)
        applyConv2dWeights(to: model.audioEncoder.conv2d3, prefix: "conv2d3", from: audioTowerWeights, transposePyTorch: true)
        CommonWeightLoader.applyLinearWeights(to: model.audioEncoder.convOut, prefix: "conv_out", from: audioTowerWeights)
        CommonWeightLoader.applyLayerNormWeights(to: model.audioEncoder.lnPost, prefix: "ln_post", from: audioTowerWeights)
        CommonWeightLoader.applyLinearWeights(to: model.audioEncoder.proj1, prefix: "proj1", from: audioTowerWeights)
        CommonWeightLoader.applyLinearWeights(to: model.audioEncoder.proj2, prefix: "proj2", from: audioTowerWeights)

        for (index, layer) in model.audioEncoder.layers.enumerated() {
            let prefix = "layers.\(index)"
            applyEncoderLayerWeights(to: layer, prefix: prefix, from: audioTowerWeights)
        }
        print("Applied audio encoder weights (\(model.audioEncoder.layers.count) layers)")

        // Load text decoder weights (quantized or float)
        if let quantized = model.textDecoder as? QuantizedTextModel {
            CommonWeightLoader.applyQuantizedEmbeddingWeights(
                to: quantized.embedTokens,
                prefix: "embed_tokens",
                from: textWeights
            )
            CommonWeightLoader.applyRMSNormWeights(to: quantized.norm, prefix: "norm", from: textWeights)

            for (index, layer) in quantized.layers.enumerated() {
                let prefix = "layers.\(index)"
                applyQuantizedDecoderLayerWeights(to: layer, prefix: prefix, from: textWeights)
            }
            print("Applied quantized text decoder weights (\(quantized.layers.count) layers)")
        } else if let floatModel = model.textDecoder as? FloatTextModel {
            CommonWeightLoader.applyEmbeddingWeights(
                to: floatModel.embedTokens,
                prefix: "embed_tokens",
                from: textWeights
            )
            CommonWeightLoader.applyRMSNormWeights(to: floatModel.norm, prefix: "norm", from: textWeights)

            for (index, layer) in floatModel.layers.enumerated() {
                let prefix = "layers.\(index)"
                applyFloatDecoderLayerWeights(to: layer, prefix: prefix, from: textWeights)
            }
            print("Applied float text decoder weights (\(floatModel.layers.count) layers)")
        }

        // Load classify head (NOT quantized — regular Linear)
        CommonWeightLoader.applyLinearWeights(to: model.classifyHead, prefix: "lm_head", from: classifyWeights)
        print("Applied classify head weights")
    }

    // MARK: - ASR-specific Weight Application Helpers

    private static func applyQuantizedDecoderLayerWeights(
        to layer: QuantizedTextDecoderLayer,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        // Self attention
        CommonWeightLoader.applyLinearWeights(to: layer.selfAttn.qProj, prefix: "\(prefix).self_attn.q_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.selfAttn.kProj, prefix: "\(prefix).self_attn.k_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.selfAttn.vProj, prefix: "\(prefix).self_attn.v_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.selfAttn.oProj, prefix: "\(prefix).self_attn.o_proj", from: weights)

        // Q/K norms
        CommonWeightLoader.applyRMSNormWeights(to: layer.selfAttn.qNorm, prefix: "\(prefix).self_attn.q_norm", from: weights)
        CommonWeightLoader.applyRMSNormWeights(to: layer.selfAttn.kNorm, prefix: "\(prefix).self_attn.k_norm", from: weights)

        // Layer norms
        CommonWeightLoader.applyRMSNormWeights(to: layer.inputLayerNorm, prefix: "\(prefix).input_layernorm", from: weights)
        CommonWeightLoader.applyRMSNormWeights(to: layer.postAttentionLayerNorm, prefix: "\(prefix).post_attention_layernorm", from: weights)

        // MLP
        CommonWeightLoader.applyLinearWeights(to: layer.mlp.gateProj, prefix: "\(prefix).mlp.gate_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.mlp.upProj, prefix: "\(prefix).mlp.up_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.mlp.downProj, prefix: "\(prefix).mlp.down_proj", from: weights)
    }

    private static func applyFloatDecoderLayerWeights(
        to layer: FloatTextDecoderLayer,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        CommonWeightLoader.applyLinearWeights(to: layer.selfAttn.qProj, prefix: "\(prefix).self_attn.q_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.selfAttn.kProj, prefix: "\(prefix).self_attn.k_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.selfAttn.vProj, prefix: "\(prefix).self_attn.v_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.selfAttn.oProj, prefix: "\(prefix).self_attn.o_proj", from: weights)
        CommonWeightLoader.applyRMSNormWeights(to: layer.selfAttn.qNorm, prefix: "\(prefix).self_attn.q_norm", from: weights)
        CommonWeightLoader.applyRMSNormWeights(to: layer.selfAttn.kNorm, prefix: "\(prefix).self_attn.k_norm", from: weights)
        CommonWeightLoader.applyRMSNormWeights(to: layer.inputLayerNorm, prefix: "\(prefix).input_layernorm", from: weights)
        CommonWeightLoader.applyRMSNormWeights(to: layer.postAttentionLayerNorm, prefix: "\(prefix).post_attention_layernorm", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.mlp.gateProj, prefix: "\(prefix).mlp.gate_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.mlp.upProj, prefix: "\(prefix).mlp.up_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.mlp.downProj, prefix: "\(prefix).mlp.down_proj", from: weights)
    }

    // MARK: - Audio Encoder Weight Helpers

    private static func applyConv2dWeights(
        to conv: Conv2d,
        prefix: String,
        from weights: [String: MLXArray],
        transposePyTorch: Bool = false
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            if transposePyTorch {
                // PyTorch Conv2d: [outC, inC, kH, kW] -> MLX Conv2d: [outC, kH, kW, inC]
                params["weight"] = .value(weight.transposed(0, 2, 3, 1))
            } else {
                params["weight"] = .value(weight)
            }
        }
        if let bias = weights["\(prefix).bias"] {
            params["bias"] = .value(bias)
        }

        if !params.isEmpty {
            conv.update(parameters: ModuleParameters(values: params))
        }
    }

    private static func applyEncoderLayerWeights(
        to layer: AudioEncoderLayer,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        // Self attention
        CommonWeightLoader.applyLinearWeights(to: layer.selfAttn.qProj, prefix: "\(prefix).self_attn.q_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.selfAttn.kProj, prefix: "\(prefix).self_attn.k_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.selfAttn.vProj, prefix: "\(prefix).self_attn.v_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.selfAttn.outProj, prefix: "\(prefix).self_attn.out_proj", from: weights)

        // Layer norms
        CommonWeightLoader.applyLayerNormWeights(to: layer.selfAttnLayerNorm, prefix: "\(prefix).self_attn_layer_norm", from: weights)
        CommonWeightLoader.applyLayerNormWeights(to: layer.finalLayerNorm, prefix: "\(prefix).final_layer_norm", from: weights)

        // FFN
        CommonWeightLoader.applyLinearWeights(to: layer.fc1, prefix: "\(prefix).fc1", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.fc2, prefix: "\(prefix).fc2", from: weights)
    }
}
