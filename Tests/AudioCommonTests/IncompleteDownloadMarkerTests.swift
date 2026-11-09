import XCTest
@testable import AudioCommon

/// An interrupted Hub download must never be served cache-first: Hub's offline
/// validation only checks files already on disk, so a partial snapshot passes it.
final class IncompleteDownloadMarkerTests: XCTestCase {
    // A repo id that does not exist on the Hub: any Hub contact fails.
    private let modelId = "speech-swift-tests/incomplete-download-marker-nonexistent"
    private var base: URL!
    private var repoDir: URL!

    override func setUpWithError() throws {
        base = FileManager.default.temporaryDirectory
            .appendingPathComponent("incomplete-marker-\(UUID().uuidString)", isDirectory: true)
        repoDir = base.appendingPathComponent("models/\(modelId)", isDirectory: true)
        try FileManager.default.createDirectory(at: repoDir, withIntermediateDirectories: true)
        // A locally valid snapshot: files plus Hub download metadata (non-LFS etag, so no hash check).
        let metadataDir = repoDir.appendingPathComponent(".cache/huggingface/download", isDirectory: true)
        try FileManager.default.createDirectory(at: metadataDir, withIntermediateDirectories: true)
        for name in ["config.json", "model.safetensors"] {
            try Data("{}".utf8).write(to: repoDir.appendingPathComponent(name))
            let metadata = "0123456789abcdef0123456789abcdef01234567\nabc123\n\(Date().timeIntervalSince1970)\n"
            try Data(metadata.utf8).write(to: metadataDir.appendingPathComponent(name + ".metadata"))
        }
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: base)
    }

    func testMarkerHelpersRoundTrip() throws {
        XCTAssertFalse(HuggingFaceDownloader.hasIncompleteDownloadMarker(in: repoDir))
        try HuggingFaceDownloader.writeIncompleteDownloadMarker(in: repoDir)
        XCTAssertTrue(HuggingFaceDownloader.hasIncompleteDownloadMarker(in: repoDir))
        HuggingFaceDownloader.removeIncompleteDownloadMarker(in: repoDir)
        XCTAssertFalse(HuggingFaceDownloader.hasIncompleteDownloadMarker(in: repoDir))
    }

    func testCompleteLocalSnapshotIsUsedWithoutHub() async throws {
        // No marker: the cache-first path accepts the snapshot and never contacts the Hub.
        try await HuggingFaceDownloader.downloadWeights(modelId: modelId, to: repoDir)
        XCTAssertFalse(HuggingFaceDownloader.hasIncompleteDownloadMarker(in: repoDir))
    }

    func testMarkedSnapshotIsNotUsedCacheFirst() async throws {
        try HuggingFaceDownloader.writeIncompleteDownloadMarker(in: repoDir)
        do {
            try await HuggingFaceDownloader.downloadWeights(modelId: modelId, to: repoDir)
            XCTFail("An interrupted download was served from the cache without the Hub")
        } catch {
            // The Hub has no such repo (or is unreachable): the download fails and the marker stays.
            XCTAssertTrue(HuggingFaceDownloader.hasIncompleteDownloadMarker(in: repoDir))
        }
    }

    func testMarkerIsHiddenFromOfflineFileListing() throws {
        // The marker must not itself look like a cached model file.
        try HuggingFaceDownloader.writeIncompleteDownloadMarker(in: repoDir)
        let visible = try FileManager.default.contentsOfDirectory(
            at: repoDir, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        XCTAssertEqual(Set(visible.map(\.lastPathComponent)), ["config.json", "model.safetensors"])
    }
}
