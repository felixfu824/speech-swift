import Foundation
import Hub
import os

/// Download errors
public enum DownloadError: Error, LocalizedError {
    case failedToDownload(String)
    case invalidRemoteFileName(String)

    public var errorDescription: String? {
        switch self {
        case .failedToDownload(let file):
            return "Failed to download: \(file)"
        case .invalidRemoteFileName(let file):
            return "Refusing to write unsafe remote file name: \(file)"
        }
    }
}

/// HuggingFace model downloader — shared between ASR, TTS, VAD, etc.
///
/// Uses `HubApi` from the swift-transformers `Hub` module for downloads,
/// which provides HF token auth, metadata tracking, and resume support.
public enum HuggingFaceDownloader {

    // MARK: - Cache Directory

    /// Get cache directory for a model.
    ///
    /// Returns the old flat cache path if it already contains model files (preserving
    /// ~10 GB of existing cached models), otherwise returns the new Hub-style path.
    public static func getCacheDirectory(for modelId: String, cacheDirName: String = "qwen3-speech") throws -> URL {
        let base = resolveBaseCacheDir(cacheDirName: cacheDirName)
        let fm = FileManager.default

        // Check old (flat) cache path for backward compat:
        //   ~/Library/Caches/qwen3-speech/aufklarer_Qwen3-ASR-0.6B-MLX-4bit/
        let oldDir = base.appendingPathComponent(sanitizedCacheKey(for: modelId), isDirectory: true)
        if weightsExist(in: oldDir) {
            return oldDir
        }

        // New Hub-style path:
        //   ~/Library/Caches/qwen3-speech/models/aufklarer/Qwen3-ASR-0.6B-MLX-4bit/
        let hub = HubApi(downloadBase: base)
        let repo = Hub.Repo(id: modelId)
        let dir = hub.localRepoLocation(repo)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    // MARK: - Weight Existence Check

    /// Check if safetensors weights exist in a directory.
    public static func weightsExist(in directory: URL) -> Bool {
        let fm = FileManager.default
        let contents: [URL]
        do {
            contents = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        } catch {
            AudioLog.download.debug("Could not list directory \(directory.path): \(error)")
            contents = []
        }
        return contents.contains { $0.pathExtension == "safetensors" }
    }

    // MARK: - Download

    /// Download model files from HuggingFace using `HubApi.snapshot()`.
    ///
    /// Builds glob patterns from the file list:
    /// - Always includes `config.json`
    /// - If `additionalFiles` doesn't contain `.safetensors` files, adds `*.safetensors`
    ///   and `model.safetensors.index.json` to discover sharded weights automatically
    /// - All entries in `additionalFiles` are added as-is (they work as glob patterns)
    public static func downloadWeights(
        modelId: String,
        to directory: URL,
        additionalFiles: [String] = [],
        progressHandler: ((Double) -> Void)? = nil
    ) async throws {
        var globs: [String] = ["config.json"]

        let hasExplicitWeights = additionalFiles.contains { $0.hasSuffix(".safetensors") }
        if !hasExplicitWeights {
            globs.append("*.safetensors")
            globs.append("model.safetensors.index.json")
        }
        for file in additionalFiles where !globs.contains(file) {
            globs.append(file)
        }

        // Derive the download base from the directory.
        // getCacheDirectory returns either:
        //   old: base/cacheKey         (flat, already has weights — won't reach here)
        //   new: base/models/org/model  (Hub-style)
        // For Hub API we need `base` as downloadBase.
        // Online only: HubApi otherwise switches itself to offline mode when
        // the network monitor reports no connection, and that branch accepts
        // whatever files are on disk. Offline reuse is handled explicitly by the
        // cache-first step below; a Hub download must really reach the Hub.
        let hub = makeHubApi(for: modelId, repoDir: directory, useOfflineMode: false)
        let repo = Hub.Repo(id: modelId)

        // Cache-first: a complete local snapshot must never require the Hub.
        // Offline mode validates the snapshot from disk (per-file metadata +
        // LFS hash integrity) without any network I/O; the online path — with
        // its per-file HEAD revalidation — is reached only when that
        // validation fails (first download, partial cache, corrupt file).
        // Without this, a Hub outage (e.g. HTTP 503 while the network itself
        // is up, which automatic offline detection does not cover) bricks
        // loading a model that is fully cached on disk.
        //
        // Offline validation only checks the files already on disk, so a
        // download interrupted between files (cancel, quit, crash, network
        // loss) would pass it with files missing. The incomplete-download
        // marker closes that gap: it exists from the start of a Hub download
        // until the download finishes, so an interrupted download always
        // resumes from the Hub. Caches without a marker (including ones written
        // before the marker existed) keep the cache-first behaviour.
        //
        // Downloads into one directory run one at a time in this process, so the
        // cache check, marker write, download, verification and marker removal
        // of one attempt never interleave with another attempt's.
        let key = directory.standardizedFileURL.path
        await directoryGate.acquire(key)
        do {
            try await downloadSerialized(
                modelId: modelId, repo: repo, globs: globs, directory: directory,
                hub: hub, progressHandler: progressHandler)
            await directoryGate.release(key)
        } catch {
            await directoryGate.release(key)
            throw error
        }
    }

    private static func downloadSerialized(
        modelId: String,
        repo: Hub.Repo,
        globs: [String],
        directory: URL,
        hub: HubApi,
        progressHandler: ((Double) -> Void)?
    ) async throws {
        try Task.checkCancellation()
        if hasIncompleteDownloadMarker(in: directory) {
            AudioLog.download.info(
                "Previous download of \(modelId, privacy: .public) did not finish; contacting Hub")
        } else {
            let offlineHub = makeHubApi(for: modelId, repoDir: directory, useOfflineMode: true)
            do {
                try await offlineHub.snapshot(from: repo, matching: globs)
                AudioLog.download.info("Using validated local cache for \(modelId, privacy: .public) (no Hub contact)")
                return
            } catch {
                AudioLog.download.info(
                    "Local cache not usable for \(modelId, privacy: .public) (\(error.localizedDescription, privacy: .public)); contacting Hub")
            }
        }

        try writeIncompleteDownloadMarker(in: directory)
        let destination: URL
        let expected: [String]
        do {
            // The file list is taken before the download, so verifying it adds
            // no request that could fail after the files are on disk. If the
            // repo changes in between, verification can only fail (keeping the
            // marker for a retry); it can never accept a partial snapshot.
            expected = try await hub.getFilenames(from: repo, matching: globs)
            destination = try await hub.snapshot(from: repo, matching: globs) { progress in
                progressHandler?(progress.fractionCompleted)
            }
        } catch {
            throw DownloadError.failedToDownload("\(modelId): \(error.localizedDescription)")
        }
        // HubApi.snapshot returns normally when its task is cancelled between
        // files; the snapshot is then incomplete, so keep the marker.
        try Task.checkCancellation()
        try verifySnapshot(modelId: modelId, expected: expected, in: destination)
        removeIncompleteDownloadMarker(in: directory)
    }

    /// Throws unless the Hub listed at least one file for the request and every
    /// listed file is on disk and non-empty. An empty list proves nothing (the
    /// repo may have removed or renamed its files), so it never clears a marker.
    static func verifySnapshot(modelId: String, expected: [String], in destination: URL) throws {
        guard !expected.isEmpty else {
            throw DownloadError.failedToDownload("\(modelId): the Hub listed no files for this model")
        }
        let missing = missingFiles(expected, in: destination)
        guard missing.isEmpty else {
            throw DownloadError.failedToDownload(
                "\(modelId): incomplete snapshot, missing \(missing.joined(separator: ", "))")
        }
    }

    // MARK: - Incomplete Download Marker

    /// Relative path of the marker that flags an unfinished Hub download.
    /// It sits under `.cache/`, which Hub's offline file listing skips.
    static let incompleteDownloadMarkerPath = ".cache/speech-swift/download-incomplete"

    /// Serializes downloads per cache directory within this process.
    static let directoryGate = DirectoryGate()

    /// Files from `expected` (Hub-relative paths) that are absent or empty under `directory`.
    static func missingFiles(_ expected: [String], in directory: URL) -> [String] {
        expected.filter { name in
            let path = directory.appendingPathComponent(name).path
            guard let size = (try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? Int else {
                return true
            }
            return size == 0
        }
    }

    /// True while a Hub download into `directory` has started but not finished
    /// (it was cancelled, interrupted, or failed). Such a cache may be missing
    /// files and is never used without contacting the Hub.
    public static func hasIncompleteDownloadMarker(in directory: URL) -> Bool {
        FileManager.default.fileExists(
            atPath: directory.appendingPathComponent(incompleteDownloadMarkerPath).path)
    }

    static func writeIncompleteDownloadMarker(in directory: URL) throws {
        let marker = directory.appendingPathComponent(incompleteDownloadMarkerPath)
        try FileManager.default.createDirectory(
            at: marker.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data().write(to: marker)
    }

    static func removeIncompleteDownloadMarker(in directory: URL) {
        let marker = directory.appendingPathComponent(incompleteDownloadMarkerPath)
        guard FileManager.default.fileExists(atPath: marker.path) else { return }
        do {
            try FileManager.default.removeItem(at: marker)
        } catch {
            // The cache stays marked (safe: the next load revalidates with the Hub).
            AudioLog.download.error(
                "Could not clear incomplete-download marker in \(directory.path, privacy: .public): \(error.localizedDescription, privacy: .public)")
        }
    }

    // MARK: - Security Helpers (kept for backward compat + security tests)

    /// Convert an arbitrary modelId into a single, safe path component for on-disk caching.
    public static func sanitizedCacheKey(for modelId: String) -> String {
        let replaced = modelId.replacingOccurrences(of: "/", with: "_")

        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        var scalars: [UnicodeScalar] = []
        scalars.reserveCapacity(replaced.unicodeScalars.count)
        for s in replaced.unicodeScalars {
            scalars.append(allowed.contains(s) ? s : "_")
        }

        var cleaned = String(String.UnicodeScalarView(scalars))
        cleaned = cleaned.trimmingCharacters(in: CharacterSet(charactersIn: "._"))

        if cleaned.isEmpty || cleaned == "." || cleaned == ".." {
            cleaned = "model"
        }

        return cleaned
    }

    /// Validate that a remote file name is safe.
    public static func validatedRemoteFileName(_ file: String) throws -> String {
        let base = URL(fileURLWithPath: file).lastPathComponent
        guard base == file else {
            throw DownloadError.invalidRemoteFileName(file)
        }
        guard !base.isEmpty, !base.hasPrefix("."), !base.contains("..") else {
            throw DownloadError.invalidRemoteFileName(file)
        }
        guard base.range(of: #"^[A-Za-z0-9._-]+$"#, options: .regularExpression) != nil else {
            throw DownloadError.invalidRemoteFileName(file)
        }
        return base
    }

    /// Validate that a local path stays within the expected directory.
    public static func validatedLocalPath(directory: URL, fileName: String) throws -> URL {
        let local = directory.appendingPathComponent(fileName, isDirectory: false)
        let dirPath = directory.standardizedFileURL.path
        let localPath = local.standardizedFileURL.path
        let prefix = dirPath.hasSuffix("/") ? dirPath : (dirPath + "/")
        guard localPath.hasPrefix(prefix) else {
            throw DownloadError.invalidRemoteFileName(fileName)
        }
        return local
    }

    // MARK: - Private Helpers

    /// Resolve the base cache directory from env vars or system default.
    private static func resolveBaseCacheDir(cacheDirName: String) -> URL {
        let fm = FileManager.default
        let root: URL
        if let override = ProcessInfo.processInfo.environment["QWEN3_CACHE_DIR"],
           !override.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            root = URL(fileURLWithPath: override, isDirectory: true)
        } else if let override = ProcessInfo.processInfo.environment["QWEN3_ASR_CACHE_DIR"],
                  !override.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            // Legacy env var support
            root = URL(fileURLWithPath: override, isDirectory: true)
        } else {
            root = fm.urls(for: .cachesDirectory, in: .userDomainMask).first!
        }
        return root.appendingPathComponent(cacheDirName, isDirectory: true)
    }

    /// Create a `HubApi` whose `downloadBase` is derived from the repo directory that
    /// `getCacheDirectory` returned (strips the `models/<org>/<model>` suffix).
    /// Test hook: point Hub requests at another host (e.g. a closed local port).
    nonisolated(unsafe) static var hubEndpointOverride: String?

    private static func makeHubApi(for modelId: String, repoDir: URL, useOfflineMode: Bool? = nil) -> HubApi {
        // repoDir is  base/models/org/model
        // We need     base
        let repo = Hub.Repo(id: modelId)
        let suffix = "/\(repo.type.rawValue)/\(repo.id)"
        let repoDirPath = repoDir.path
        let downloadBase: URL
        if repoDirPath.hasSuffix(suffix) {
            let basePath = String(repoDirPath.dropLast(suffix.count))
            downloadBase = URL(fileURLWithPath: basePath, isDirectory: true)
        } else {
            // Fallback: old-style flat dir — use its parent as downloadBase.
            // Hub won't match this path, so we derive base from env/defaults.
            downloadBase = resolveBaseCacheDir(cacheDirName: repoDir.deletingLastPathComponent().lastPathComponent)
        }
        return HubApi(downloadBase: downloadBase, endpoint: hubEndpointOverride, useOfflineMode: useOfflineMode)
    }
}

/// An async mutex per key: one holder at a time, waiters served in order.
/// Ownership passes directly from a releasing holder to the next waiter.
actor DirectoryGate {
    private var held: Set<String> = []
    private var waiters: [String: [CheckedContinuation<Void, Never>]] = [:]

    func acquire(_ key: String) async {
        guard held.contains(key) else {
            held.insert(key)
            return
        }
        await withCheckedContinuation { waiters[key, default: []].append($0) }
    }

    func release(_ key: String) {
        if var queue = waiters[key], !queue.isEmpty {
            let next = queue.removeFirst()
            waiters[key] = queue.isEmpty ? nil : queue
            next.resume()
        } else {
            held.remove(key)
        }
    }

    func isHeld(_ key: String) -> Bool { held.contains(key) }
}
