import XCTest
import Foundation
import AVFoundation
import Darwin
@testable import Qwen3ASR

/// Opt-in offline runner; input/output locations are supplied by the caller.
final class E2EMixedPrecisionParityTests: XCTestCase {
    func testOfflineParity() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let modelPath = env["ASR_PARITY_MODEL"], let input = env["ASR_PARITY_INPUT"],
              let output = env["ASR_PARITY_OUTPUT"] else { throw XCTSkip("Offline parity paths not supplied") }
        let start = Date()
        let model = try await Qwen3ASRModel.fromPretrained(modelId: modelPath)
        // ASR_PARITY_AUTO_LANGUAGE_PROMPT=1 reproduces mlx-audio's auto-language prompt.
        model.usesAutoLanguagePrompt = env["ASR_PARITY_AUTO_LANGUAGE_PROMPT"] == "1"
        let loadTime = Date().timeIntervalSince(start)
        let lines = try String(contentsOfFile: input, encoding: .utf8).split(separator: "\n")
        FileManager.default.createFile(atPath: output, contents: nil)
        let handle = try FileHandle(forWritingTo: URL(fileURLWithPath: output))
        defer { try? handle.close() }
        for line in lines.prefix(Int(env["ASR_PARITY_LIMIT"] ?? "9999") ?? 9999) {
            var row: [String: Any]
            if line.first == "{" { row = try JSONSerialization.jsonObject(with: Data(line.utf8)) as! [String: Any] }
            else { row = ["wav": String(line)] }
            let file = try AVAudioFile(forReading: URL(fileURLWithPath: row["wav"] as! String))
            let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
            try file.read(into: buffer)
            let samples = Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
            row["swift_text"] = model.transcribe(audio: samples, sampleRate: Int(file.processingFormat.sampleRate), language: env["ASR_PARITY_LANGUAGE"], maxTokens: 256)
            try handle.write(contentsOf: JSONSerialization.data(withJSONObject: row, options: [.sortedKeys]))
            try handle.write(contentsOf: Data([10]))
            print("PARITY \(row["wav"]!): \(row["swift_text"]!)")
        }
        var usage = rusage()
        getrusage(RUSAGE_SELF, &usage)
        let stats: [String: Any] = ["load_seconds": loadTime, "peak_rss_bytes": usage.ru_maxrss]
        try JSONSerialization.data(withJSONObject: stats, options: [.prettyPrinted]).write(to: URL(fileURLWithPath: output + ".stats.json"))
    }
}
