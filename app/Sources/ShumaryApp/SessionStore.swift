import Foundation

/// One recording or import, as it exists on disk.
///
/// `metadata.json` is the single source of truth for what still needs doing —
/// the queue is rebuilt from it at launch rather than kept only in memory, so a
/// crash mid-transcription costs nothing.
struct SessionMetadata: Codable {
    enum State: String, Codable {
        case recording      // capture in progress; nothing to queue yet
        case pending        // audio on disk, waiting for the engine
        case transcribing   // engine running; on next launch this means it died
        case done
        case failed
    }

    enum Source: String, Codable {
        case recorded
        case imported
    }

    var id: String
    var source: Source
    var state: State
    var startedAt: String
    var audioFile: String
    var dualSource: Bool
    var durationSec: Double?
    var tracks: [String: String]?
    var speakers: [String]?
    var originalFilename: String?
    var error: String?
    var conversion: String?
    var language: String?

    enum CodingKeys: String, CodingKey {
        case id, source, state, tracks, speakers, error, conversion, language
        case startedAt = "started_at"
        case audioFile = "audio_file"
        case dualSource = "dual_source"
        case durationSec = "duration_sec"
        case originalFilename = "original_filename"
    }
}

enum SessionStore {
    static let metadataName = "metadata.json"

    static func metadataURL(in folder: URL) -> URL {
        folder.appendingPathComponent(metadataName)
    }

    static func load(from folder: URL) -> SessionMetadata? {
        guard let data = try? Data(contentsOf: metadataURL(in: folder)) else { return nil }
        return try? JSONDecoder().decode(SessionMetadata.self, from: data)
    }

    static func save(_ metadata: SessionMetadata, to folder: URL) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(metadata) else { return }
        try? data.write(to: metadataURL(in: folder), options: .atomic)
    }

    /// Mutate a session's metadata in place.
    static func update(_ folder: URL, _ change: (inout SessionMetadata) -> Void) {
        guard var metadata = load(from: folder) else { return }
        change(&metadata)
        save(metadata, to: folder)
    }

    static func audioURL(in folder: URL) -> URL? {
        guard let metadata = load(from: folder) else { return nil }
        let url = folder.appendingPathComponent(metadata.audioFile)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// Sessions still owing a transcript, oldest first.
    ///
    /// `transcribing` counts as outstanding: seeing it at launch means a
    /// previous run died part-way, and the raw audio is still there to retry.
    static func outstanding() -> [URL] {
        Paths.allSessions().filter { folder in
            guard let metadata = load(from: folder) else { return false }
            return metadata.state == .pending || metadata.state == .transcribing
        }
    }
}
