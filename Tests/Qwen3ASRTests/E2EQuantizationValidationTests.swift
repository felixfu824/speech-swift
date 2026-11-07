import XCTest
import MLX
import MLXNN
import MLXCommon
@testable import Qwen3ASR

/// Uses MLX allocations; excluded from the network-free config-only suite by the E2E prefix.
final class E2EQuantizationValidationTests: XCTestCase {
    func testStrictMissingExtraAndShapeMismatch() throws {
        let layer = Linear(64, 32, bias: false)
        let valid = ["test.weight": MLXArray.zeros([32, 64])]
        XCTAssertNoThrow(try WeightLoader.applyStrict(layer, weights: valid, prefix: "test."))
        XCTAssertThrowsError(try WeightLoader.applyStrict(layer, weights: [:], prefix: "test."))
        XCTAssertThrowsError(try WeightLoader.applyStrict(layer, weights: ["test.weight": MLXArray.zeros([32, 32])], prefix: "test."))
        XCTAssertThrowsError(try WeightLoader.applyStrict(layer, weights: valid.merging(["test.unexpected": MLXArray.zeros([1])]) { $1 }, prefix: "test."))
    }
    func testSixBitEmbeddingPackedShape() {
        let embedding = PreQuantizedEmbedding(embeddingCount: 128, dimensions: 1024, bits: 6)
        XCTAssertEqual(embedding.weight.shape, [128, 192])
    }
    func testFloatOverrideAndEmbedding() throws {
        var config = TextDecoderConfig.small
        config.hiddenSize = 64
        config.vocabSize = 128
        config.numLayers = 0
        config.quantization = try JSONDecoder().decode(ASRQuantization.self, from: Data(#"{"group_size":64,"bits":6,"model.embed_tokens":false,"model.test":false}"#.utf8))
        XCTAssertFalse(config.makeLinear(64, 32, bias: false, path: "model.test") is QuantizedLinear)
        let model = QuantizedTextModel(config: config)
        XCTAssertEqual(model.embedTokens.bits, 0)
        XCTAssertEqual(model.embedTokens.weight.shape, [128, 64])
        try WeightLoader.applyStrict(model, weights: ["model.embed_tokens.weight": MLXArray.zeros([128, 64]), "model.norm.weight": MLXArray.ones([64])], prefix: "model.")
        XCTAssertEqual(model.embedTokens(MLXArray([Int32(0)])).shape, [1, 64])
    }
}
