import Foundation
import MLXCommon
import MLX
import MLXNN
import MLXFast
import AudioCommon

/// Multi-head attention for Qwen3 text decoder with GQA and RoPE (quantized version)
public class QuantizedTextAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: MLXNN.RoPE

    public convenience init(config: TextDecoderConfig) {
        self.init(config: config, path: "")
    }

    public init(config: TextDecoderConfig, path: String) {
        self.numHeads = config.numHeads
        self.numKVHeads = config.numKVHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(headDim))

        let hiddenSize = config.hiddenSize

        // Create quantized linear layers
        self._qProj.wrappedValue = config.makeLinear(
            hiddenSize, numHeads * headDim, bias: false,
            path: path + ".q_proj")
        self._kProj.wrappedValue = config.makeLinear(
            hiddenSize, numKVHeads * headDim, bias: false,
            path: path + ".k_proj")
        self._vProj.wrappedValue = config.makeLinear(
            hiddenSize, numKVHeads * headDim, bias: false,
            path: path + ".v_proj")
        self._oProj.wrappedValue = config.makeLinear(
            numHeads * headDim, hiddenSize, bias: false,
            path: path + ".o_proj")

        // Q/K normalization (Qwen3 specific)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        // MLXFast RoPE: split-half rotation (traditional=false), base from config
        self.rope = MLXNN.RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)

        super.init()
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil,
        cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (batch, seqLen, _) = (hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2))

        // Project Q, K, V
        var queries = qProj(hiddenStates)
        var keys = kProj(hiddenStates)
        var values = vProj(hiddenStates)

        // Reshape for multi-head attention
        queries = queries.reshaped(batch, seqLen, numHeads, headDim)
        keys = keys.reshaped(batch, seqLen, numKVHeads, headDim)
        values = values.reshaped(batch, seqLen, numKVHeads, headDim)

        // Apply Q/K normalization
        queries = qNorm(queries)
        keys = kNorm(keys)

        // Transpose to [batch, heads, seq, head_dim]
        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        // Calculate offset for RoPE based on cache
        let offset = cache?.0.dim(2) ?? 0

        // Apply MLXFast RoPE (handles split-half rotation via optimized Metal kernel)
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        // Update cache
        var cachedKeys = keys
        var cachedValues = values

        if let (prevKeys, prevValues) = cache {
            cachedKeys = concatenated([prevKeys, keys], axis: 2)
            cachedValues = concatenated([prevValues, values], axis: 2)
        }

        // SDPA handles GQA natively (N_q != N_kv), no need to tile KV heads
        let attnOutput = MLXFast.scaledDotProductAttention(
            queries: queries, keys: cachedKeys, values: cachedValues,
            scale: scale, mask: attentionMask)

        // SDPA returns [B, N_q, T_q, D], transpose to [B, T_q, N_q, D] then reshape
        let output = oProj(attnOutput.transposed(0, 2, 1, 3).reshaped(batch, seqLen, numHeads * headDim))

        return (output, (cachedKeys, cachedValues))
    }
}

/// MLP for Qwen3 text decoder (SwiGLU activation, quantized)
/// Wraps the shared QuantizedMLP for backward compatibility
public class QuantizedTextMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    public convenience init(config: TextDecoderConfig) {
        self.init(config: config, path: "")
    }

    public init(config: TextDecoderConfig, path: String) {
        let hiddenSize = config.hiddenSize
        let intermediateSize = config.intermediateSize

        self._gateProj.wrappedValue = config.makeLinear(
            hiddenSize, intermediateSize, bias: false,
            path: path + ".gate_proj")
        self._upProj.wrappedValue = config.makeLinear(
            hiddenSize, intermediateSize, bias: false,
            path: path + ".up_proj")
        self._downProj.wrappedValue = config.makeLinear(
            intermediateSize, hiddenSize, bias: false,
            path: path + ".down_proj")

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // SwiGLU: down(silu(gate(x)) * up(x))
        let gate = silu(gateProj(x))
        let up = upProj(x)
        return downProj(gate * up)
    }
}

/// Decoder layer for Qwen3 text model (quantized)
public class QuantizedTextDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: QuantizedTextAttention
    @ModuleInfo var mlp: QuantizedTextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public convenience init(config: TextDecoderConfig) {
        self.init(config: config, path: "")
    }

    public init(config: TextDecoderConfig, path: String) {
        self._selfAttn.wrappedValue = QuantizedTextAttention(config: config, path: path + ".self_attn")
        self._mlp.wrappedValue = QuantizedTextMLP(config: config, path: path + ".mlp")
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        attentionMask: MLXArray? = nil,
        cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        // Self attention with pre-norm
        let residual = hiddenStates
        var hidden = inputLayerNorm(hiddenStates)
        let (attnOutput, newCache) = selfAttn(hidden, attentionMask: attentionMask, cache: cache)
        hidden = residual + attnOutput

        // MLP with pre-norm
        let residual2 = hidden
        hidden = postAttentionLayerNorm(hidden)
        hidden = mlp(hidden)
        hidden = residual2 + hidden

        return (hidden, newCache)
    }
}

/// Full Qwen3 text decoder model (quantized)
public class QuantizedTextModel: Module {
    public let config: TextDecoderConfig

    @ModuleInfo(key: "embed_tokens") public var embedTokens: PreQuantizedEmbedding
    @ModuleInfo var layers: [QuantizedTextDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    public init(config: TextDecoderConfig) {
        self.config = config

        self._embedTokens.wrappedValue = PreQuantizedEmbedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize,
            groupSize: config.embeddingSettings.groupSize,
            bits: config.embeddingSettings.bits)
        self._layers.wrappedValue = (0..<config.numLayers).map { i in
            QuantizedTextDecoderLayer(config: config, path: "model.layers.\(i)")
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    /// Forward pass through text decoder
    public func callAsFunction(
        inputIds: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        attentionMask: MLXArray? = nil,
        cache: [(MLXArray, MLXArray)]? = nil
    ) -> (MLXArray, [(MLXArray, MLXArray)]) {
        // Get embeddings
        var hiddenStates: MLXArray
        if let embeds = inputsEmbeds {
            hiddenStates = embeds
        } else if let ids = inputIds {
            hiddenStates = embedTokens(ids)
        } else {
            fatalError("Either inputIds or inputsEmbeds must be provided")
        }

        let seqLen = hiddenStates.dim(1)

        // Determine attention mask
        let mask: MLXArray?
        if let providedMask = attentionMask {
            mask = providedMask
        } else if seqLen == 1 {
            // Autoregressive: single query can attend to all cached positions, no mask needed
            mask = nil
        } else {
            // Prefill: create causal mask using MLX broadcast operations
            let cacheLen = cache?.first?.0.dim(2) ?? 0
            let totalLen = seqLen + cacheLen
            let rows = (MLXArray(0..<Int32(seqLen)) + Int32(cacheLen)).expandedDimensions(axis: 1)
            let cols = MLXArray(0..<Int32(totalLen)).expandedDimensions(axis: 0)
            mask = MLX.where(cols .> rows, MLXArray(Float(-1e9)), MLXArray(Float(0)))
                .expandedDimensions(axes: [0, 1])
                .asType(hiddenStates.dtype)
        }

        // Apply decoder layers
        var newCache: [(MLXArray, MLXArray)] = []
        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            let (output, updatedCache) = layer(hiddenStates, attentionMask: mask, cache: layerCache)
            hiddenStates = output
            newCache.append(updatedCache)
        }

        // Final norm
        hiddenStates = norm(hiddenStates)

        return (hiddenStates, newCache)
    }
}
