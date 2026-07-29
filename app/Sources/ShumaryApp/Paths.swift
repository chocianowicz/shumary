import Foundation

/// Everything shumary writes lives under one visible folder the user owns.
enum Paths {
    static let root = FileManager.default
        .homeDirectoryForCurrentUser
        .appendingPathComponent("Documents/shumary", isDirectory: true)

    static let inbox = root.appendingPathComponent("Inbox", isDirectory: true)
    static let recordings = root.appendingPathComponent("Recordings", isDirectory: true)

    static func ensure() {
        for dir in [root, inbox, recordings] {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
    }

    /// A fresh `yyyy-MM-dd-HHmm` session folder, suffixed if that minute is taken.
    static func newSessionFolder(now: Date = Date()) throws -> URL {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd-HHmm"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        let base = formatter.string(from: now)

        var candidate = recordings.appendingPathComponent(base, isDirectory: true)
        var suffix = 2
        while FileManager.default.fileExists(atPath: candidate.path) {
            candidate = recordings.appendingPathComponent("\(base)-\(suffix)", isDirectory: true)
            suffix += 1
        }
        try FileManager.default.createDirectory(at: candidate, withIntermediateDirectories: true)
        return candidate
    }

    /// Session folders, newest last.
    static func allSessions() -> [URL] {
        let contents = (try? FileManager.default.contentsOfDirectory(
            at: recordings, includingPropertiesForKeys: [.isDirectoryKey])) ?? []
        return contents
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }
}
