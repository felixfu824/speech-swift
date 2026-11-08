import XCTest
@testable import Qwen3ASR

final class MixedPrecisionConfigTests: XCTestCase {
    func testDefaultsOverridesAndFloat() throws {
        let json = #"{"group_size":64,"bits":6,"model.layers.0.self_attn.q_proj":{"group_size":32,"bits":8},"audio_tower.proj1":false}"#
        let recipe = try JSONDecoder().decode(ASRQuantization.self, from: Data(json.utf8))
        var config = TextDecoderConfig.small
        config.quantization = recipe
        XCTAssertNotNil(config.quantization)
        XCTAssertEqual(recipe.settings(for: "model.embed_tokens")?.bits, 6)
        XCTAssertEqual(recipe.settings(for: "model.layers.0.self_attn.q_proj")?.groupSize, 32)
        XCTAssertNil(recipe.settings(for: "audio_tower.proj1"))
        let copy = try JSONDecoder().decode(ASRQuantization.self, from: JSONEncoder().encode(recipe))
        XCTAssertNil(copy.settings(for: "audio_tower.proj1"))
    }
    func testOriginalInitializerReferencesRemainAvailable() {
        let _: (TextDecoderConfig) -> QuantizedTextModel = QuantizedTextModel.init(config:)
        let _: (TextDecoderConfig) -> QuantizedTextAttention = QuantizedTextAttention.init(config:)
        let _: (TextDecoderConfig) -> QuantizedTextMLP = QuantizedTextMLP.init(config:)
        let _: (TextDecoderConfig) -> QuantizedTextDecoderLayer = QuantizedTextDecoderLayer.init(config:)
    }
    func testInvalidRecipeFails() {
        for json in [#"{"group_size":64,"bits":5}"#, #"{"group_size":64,"bits":6,"mode":"unknown"}"#, #"{"group_size":64,"bits":6,"model.embed_tokens":true}"#] {
            XCTAssertThrowsError(try JSONDecoder().decode(ASRQuantization.self, from: Data(json.utf8)))
        }
    }
}
