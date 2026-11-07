import Foundation
import MLX
import MLXNN

/// Pre-quantized embedding that can be loaded directly from safetensors
public class PreQuantizedEmbedding: Module {
    public let groupSize: Int
    public let bits: Int
    public let embeddingCount: Int
    public let dimensions: Int

    @ParameterInfo public var weight: MLXArray
    @ParameterInfo public var scales: MLXArray
    @ParameterInfo public var biases: MLXArray

    public init(embeddingCount: Int, dimensions: Int, groupSize: Int = 64, bits: Int = 4) {
        self.embeddingCount = embeddingCount
        self.dimensions = dimensions
        self.groupSize = groupSize
        self.bits = bits

        // Packed dimensions: 8 values per uint32 for 4-bit
        let packedDim = bits == 0 ? dimensions : dimensions * bits / 32
        let numGroups = dimensions / groupSize

        // Initialize with zeros - will be loaded from weights
        self._weight.wrappedValue = MLXArray.zeros([embeddingCount, packedDim], dtype: bits == 0 ? .bfloat16 : .uint32)
        self._scales.wrappedValue = MLXArray.zeros([embeddingCount, numGroups], dtype: .bfloat16)
        self._biases.wrappedValue = MLXArray.zeros([embeddingCount, numGroups], dtype: .bfloat16)

        super.init()
        self.freeze()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if bits == 0 { return weight[x] }
        let s = x.shape
        let x = x.flattened()
        let out = dequantized(
            weight[x], scales: scales[x], biases: biases[x],
            groupSize: groupSize, bits: bits)
        return out.reshaped(s + [-1])
    }

    /// For use as LM head (matmul with transposed weight)
    public func asLinear(_ x: MLXArray) -> MLXArray {
        if bits == 0 { return matmul(x, weight.T) }
        return quantizedMatmul(
            x, weight, scales: scales, biases: biases, transpose: true,
            groupSize: groupSize, bits: bits)
    }
}
