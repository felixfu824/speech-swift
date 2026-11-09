import XCTest
@testable import AudioCommon

/// An interrupted Hub download must never be served from disk as complete: Hub's
/// offline validation only checks the files already present, so a partial
/// snapshot passes it. All tests are offline and deterministic: Hub requests go
/// to a closed local port.
final class IncompleteDownloadMarkerTests: XCTestCase {
    private let modelId = "speech-swift-tests/incomplete-download-marker"
    private var base: URL!
    private var repoDir: URL!

    override func setUpWithError() throws {
        HuggingFaceDownloader.hubEndpointOverride = "http://127.0.0.1:9"
        base = FileManager.default.temporaryDirectory
            .appendingPathComponent("incomplete-marker-\(UUID().uuidString)", isDirectory: true)
        repoDir = base.appendingPathComponent("models/\(modelId)", isDirectory: true)
        try FileManager.default.createDirectory(at: repoDir, withIntermediateDirectories: true)
        // A partial snapshot that is locally valid: files plus Hub download metadata
        // (non-LFS etag, so no hash check). The requested vocab.json is missing.
        let metadataDir = repoDir.appendingPathComponent(".cache/huggingface/download", isDirectory: true)
        try FileManager.default.createDirectory(at: metadataDir, withIntermediateDirectories: true)
        for name in ["config.json", "model.safetensors"] {
            try Data("{}".utf8).write(to: repoDir.appendingPathComponent(name))
            let metadata = "0123456789abcdef0123456789abcdef01234567\nabc123\n\(Date().timeIntervalSince1970)\n"
            try Data(metadata.utf8).write(to: metadataDir.appendingPathComponent(name + ".metadata"))
        }
    }

    override func tearDownWithError() throws {
        HuggingFaceDownloader.hubEndpointOverride = nil
        try? FileManager.default.removeItem(at: base)
    }

    private var marked: Bool { HuggingFaceDownloader.hasIncompleteDownloadMarker(in: repoDir) }

    func testMarkerHelpersRoundTrip() throws {
        XCTAssertFalse(marked)
        try HuggingFaceDownloader.writeIncompleteDownloadMarker(in: repoDir)
        XCTAssertTrue(marked)
        HuggingFaceDownloader.removeIncompleteDownloadMarker(in: repoDir)
        XCTAssertFalse(marked)
    }

    func testUnmarkedLocalSnapshotIsUsedWithoutHub() async throws {
        // Unchanged cache-first behaviour: no marker, Hub unreachable, still loads from disk.
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId, to: repoDir, additionalFiles: ["vocab.json"])
        XCTAssertFalse(marked)
    }

    func testMarkedPartialSnapshotNeedsTheHubAndKeepsItsMarker() async throws {
        // The blocker case: a marked partial cache with no reachable Hub must fail
        // (Hub must not fall back to its own offline branch) and stay marked.
        try HuggingFaceDownloader.writeIncompleteDownloadMarker(in: repoDir)
        do {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: modelId, to: repoDir, additionalFiles: ["vocab.json"])
            XCTFail("A marked partial snapshot loaded without the Hub")
        } catch {
            XCTAssertTrue(marked, "a failed resume must keep the marker")
        }
        // And again: the marker survives repeated failures.
        _ = try? await HuggingFaceDownloader.downloadWeights(
            modelId: modelId, to: repoDir, additionalFiles: ["vocab.json"])
        XCTAssertTrue(marked)
    }

    func testFirstDownloadFailureLeavesAMarker() async throws {
        // A download that never completes (Hub unreachable) marks the directory.
        let fresh = base.appendingPathComponent("models/speech-swift-tests/fresh", isDirectory: true)
        try FileManager.default.createDirectory(at: fresh, withIntermediateDirectories: true)
        _ = try? await HuggingFaceDownloader.downloadWeights(modelId: "speech-swift-tests/fresh", to: fresh)
        XCTAssertTrue(HuggingFaceDownloader.hasIncompleteDownloadMarker(in: fresh))
    }

    func testMissingFilesListsAbsentAndEmptyFiles() throws {
        try Data().write(to: repoDir.appendingPathComponent("empty.json"))
        let missing = HuggingFaceDownloader.missingFiles(
            ["config.json", "model.safetensors", "vocab.json", "empty.json"], in: repoDir)
        XCTAssertEqual(missing, ["vocab.json", "empty.json"])
    }

    func testMarkerClearsOnlyAfterTheLastOverlappingAttempt() throws {
        try HuggingFaceDownloader.writeIncompleteDownloadMarker(in: repoDir)
        // Two overlapping attempts; the verified one ends first: the other is still running.
        HuggingFaceDownloader.beginDownloadAttempt(in: repoDir)
        HuggingFaceDownloader.beginDownloadAttempt(in: repoDir)
        HuggingFaceDownloader.endDownloadAttempt(in: repoDir, verifiedComplete: true)
        XCTAssertTrue(marked, "a concurrent attempt is still running")
        HuggingFaceDownloader.endDownloadAttempt(in: repoDir, verifiedComplete: false)
        XCTAssertFalse(marked, "the group had a verified complete snapshot")

        // A lone attempt that did not verify keeps the marker.
        try HuggingFaceDownloader.writeIncompleteDownloadMarker(in: repoDir)
        HuggingFaceDownloader.beginDownloadAttempt(in: repoDir)
        HuggingFaceDownloader.endDownloadAttempt(in: repoDir, verifiedComplete: false)
        XCTAssertTrue(marked)
    }

    func testMarkerIsHiddenFromOfflineFileListing() throws {
        // The marker must not itself look like a cached model file.
        try HuggingFaceDownloader.writeIncompleteDownloadMarker(in: repoDir)
        let visible = try FileManager.default.contentsOfDirectory(
            at: repoDir, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        XCTAssertEqual(Set(visible.map(\.lastPathComponent)), ["config.json", "model.safetensors"])
    }
}
