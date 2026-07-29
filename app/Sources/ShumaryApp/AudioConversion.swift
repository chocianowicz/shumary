import Foundation

/// Shrinking a finished recording: 48 kHz stereo WAV is ~690 MB/hour, Opus is
/// ~29 MB/hour and loses nothing that matters for speech.
///
/// This only ever runs *after* a transcript exists, and the WAV is only deleted
/// once both replacements are verified — raw audio is the one thing that cannot
/// be regenerated.
enum AudioConversion {

    enum Result {
        case converted(mic: String, system: String)
        case skipped(String)
    }

    static func ffmpegPath() -> String? {
        ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"]
            .first { FileManager.default.isExecutableFile(atPath: $0) }
    }

    /// Split a dual-source WAV into two Opus tracks and remove the WAV.
    static func splitAndCompress(wav: URL, in folder: URL) -> Result {
        guard let ffmpeg = ffmpegPath() else {
            return .skipped("ffmpeg not installed")
        }

        let mic = folder.appendingPathComponent("mic.opus")
        let system = folder.appendingPathComponent("system.opus")

        func discardPartialOutput() {
            try? FileManager.default.removeItem(at: mic)
            try? FileManager.default.removeItem(at: system)
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: ffmpeg)
        process.arguments = [
            "-y", "-i", wav.path,
            // `aformat` is not decorative: channelsplit labels its outputs FL
            // and FR, and libopus refuses a single channel carrying a stereo
            // position ("Invalid channel layout 1 channels (FL)"). Relabelling
            // each side as plain mono is what makes the encode legal.
            "-filter_complex",
            "[0:a]channelsplit=channel_layout=stereo[l0][r0];"
                + "[l0]aformat=channel_layouts=mono[l];"
                + "[r0]aformat=channel_layouts=mono[r]",
            "-map", "[l]", "-c:a", "libopus", "-b:a", "32k", mic.path,
            "-map", "[r]", "-c:a", "libopus", "-b:a", "32k", system.path,
        ]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice

        do {
            try process.run()
        } catch {
            discardPartialOutput()
            return .skipped("ffmpeg failed to start")
        }
        process.waitUntilExit()

        guard process.terminationStatus == 0,
              FileManager.default.fileExists(atPath: mic.path),
              FileManager.default.fileExists(atPath: system.path)
        else {
            // ffmpeg leaves empty files behind when an encoder refuses to open.
            discardPartialOutput()
            return .skipped("ffmpeg conversion failed")
        }

        // Only drop the source once both outputs are real and the right length.
        guard let original = duration(of: wav, ffmpeg: ffmpeg),
              let micDuration = duration(of: mic, ffmpeg: ffmpeg),
              let systemDuration = duration(of: system, ffmpeg: ffmpeg),
              abs(micDuration - original) < 0.5,
              abs(systemDuration - original) < 0.5
        else {
            discardPartialOutput()
            return .skipped("converted tracks did not match the original duration")
        }

        try? FileManager.default.removeItem(at: wav)
        return .converted(mic: mic.lastPathComponent, system: system.lastPathComponent)
    }

    private static func duration(of url: URL, ffmpeg: String) -> Double? {
        let ffprobe = ffmpeg.replacingOccurrences(of: "ffmpeg", with: "ffprobe")
        guard FileManager.default.isExecutableFile(atPath: ffprobe) else { return nil }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: ffprobe)
        process.arguments = [
            "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", url.path,
        ]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice

        guard (try? process.run()) != nil else { return nil }
        process.waitUntilExit()
        let text = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return Double(text)
    }
}
