import Foundation
import MLXNN
import MLXCommon

/// Quantization defaults and checkpoint-path overrides. A false override keeps a module floating point.
public struct ASRQuantization: Codable, Sendable {
    public struct Settings: Codable, Sendable {
        public var groupSize: Int
        public var bits: Int
        enum CodingKeys: String, CodingKey { case groupSize = "group_size", bits }
    }
    public enum Override: Codable, Sendable {
        case floatingPoint
        case quantized(Settings)
        public init(from decoder: Decoder) throws {
            let value = try decoder.singleValueContainer()
            if let flag = try? value.decode(Bool.self), flag == false { self = .floatingPoint }
            else { self = .quantized(try value.decode(Settings.self)) }
        }
        public func encode(to encoder: Encoder) throws {
            var value = encoder.singleValueContainer()
            switch self {
            case .floatingPoint: try value.encode(false)
            case .quantized(let settings): try value.encode(settings)
            }
        }
    }
    public var defaults: Settings
    public var overrides: [String: Override]
    struct Key: CodingKey {
        var stringValue: String
        var intValue: Int? { nil }
        init(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
    }
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: Key.self)
        defaults = Settings(groupSize: try values.decode(Int.self, forKey: Key(stringValue: "group_size")),
                            bits: try values.decode(Int.self, forKey: Key(stringValue: "bits")))
        let mode = try values.decodeIfPresent(String.self, forKey: Key(stringValue: "mode")) ?? "affine"
        guard mode == "affine" else { throw WeightLoadingError.incompatibleWeights("Unsupported quantization mode: \(mode)") }
        overrides = [:]
        for key in values.allKeys where !["group_size", "bits", "mode"].contains(key.stringValue) {
            let path = key.stringValue.hasPrefix("thinker.") ? String(key.stringValue.dropFirst(8)) : key.stringValue
            overrides[path] = try values.decode(Override.self, forKey: key)
        }
        for settings in [defaults] + overrides.values.compactMap({ if case .quantized(let s) = $0 { return s }; return nil }) {
            guard [2, 3, 4, 6, 8].contains(settings.bits), [32, 64, 128].contains(settings.groupSize) else {
                throw WeightLoadingError.incompatibleWeights("Invalid quantization bits/group size")
            }
        }
    }
    public func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: Key.self)
        try values.encode(defaults.groupSize, forKey: Key(stringValue: "group_size"))
        try values.encode(defaults.bits, forKey: Key(stringValue: "bits"))
        try values.encode("affine", forKey: Key(stringValue: "mode"))
        for (key, value) in overrides { try values.encode(value, forKey: Key(stringValue: key)) }
    }
    public func settings(for path: String) -> Settings? {
        switch overrides[path] {
        case .floatingPoint: return nil
        case .quantized(let settings): return settings
        case nil: return defaults
        }
    }
}

extension TextDecoderConfig {
    // Decoder-only global recipes retain the historical aufklarer prompting.
    var usesLegacyASRPrompt: Bool { quantization?.overrides.isEmpty != false }

    var embeddingSettings: ASRQuantization.Settings {
        guard let quantization else { return .init(groupSize: groupSize, bits: bits) }
        return quantization.settings(for: "model.embed_tokens") ?? .init(groupSize: groupSize, bits: 0)
    }
    func makeLinear(_ input: Int, _ output: Int, bias: Bool, path: String) -> Linear {
        let settings: ASRQuantization.Settings?
        if let quantization { settings = quantization.settings(for: path) }
        else { settings = .init(groupSize: groupSize, bits: bits) }
        if let settings {
            return QuantizedLinear(input, output, bias: bias, groupSize: settings.groupSize, bits: settings.bits)
        }
        return Linear(input, output, bias: bias)
    }
}
