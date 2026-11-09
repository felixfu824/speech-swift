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

    func testConcurrentResumesOfAMarkedCacheAllFailAndKeepTheMarker() async throws {
        try HuggingFaceDownloader.writeIncompleteDownloadMarker(in: repoDir)
        let dir = repoDir!, id = modelId
        let failures = await withTaskGroup(of: Bool.self) { group -> Int in
            for _ in 0..<4 {
                group.addTask {
                    do {
                        try await HuggingFaceDownloader.downloadWeights(
                            modelId: id, to: dir, additionalFiles: ["vocab.json"])
                        return false
                    } catch { return true }
                }
            }
            return await group.reduce(0) { $0 + ($1 ? 1 : 0) }
        }
        XCTAssertEqual(failures, 4)
        XCTAssertTrue(marked)
        let held = await HuggingFaceDownloader.directoryGate.isHeld(repoDir.standardizedFileURL.path)
        XCTAssertFalse(held, "every attempt released the directory gate")
    }

    func testDirectoryGateAdmitsOneHolderAtATime() async throws {
        let gate = DirectoryGate()
        await gate.acquire("k")
        let order = OrderLog()
        let waiter = Task {
            await gate.acquire("k")
            await order.append("waiter")
            await gate.release("k")
        }
        try await Task.sleep(nanoseconds: 50_000_000)
        await order.append("holder")
        await gate.release("k")
        await waiter.value
        let events = await order.events
        XCTAssertEqual(events, ["holder", "waiter"])
        let held = await gate.isHeld("k")
        XCTAssertFalse(held)
    }

    func testVerificationRejectsAnEmptyHubFileList() throws {
        // An empty manifest must never count as a complete snapshot.
        XCTAssertThrowsError(try HuggingFaceDownloader.verifySnapshot(modelId: modelId, expected: [], in: repoDir))
        XCTAssertThrowsError(try HuggingFaceDownloader.verifySnapshot(
            modelId: modelId, expected: ["config.json", "vocab.json"], in: repoDir))
        XCTAssertNoThrow(try HuggingFaceDownloader.verifySnapshot(
            modelId: modelId, expected: ["config.json", "model.safetensors"], in: repoDir))
    }

    func testMarkerIsHiddenFromOfflineFileListing() throws {
        // The marker must not itself look like a cached model file.
        try HuggingFaceDownloader.writeIncompleteDownloadMarker(in: repoDir)
        let visible = try FileManager.default.contentsOfDirectory(
            at: repoDir, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        XCTAssertEqual(Set(visible.map(\.lastPathComponent)), ["config.json", "model.safetensors"])
    }
}

private actor OrderLog {
    var events: [String] = []
    func append(_ e: String) { events.append(e) }
}
