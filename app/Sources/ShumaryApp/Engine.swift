import Foundation

/// Bridge to the `shumary` Python CLI, which does the actual transcription.
///
/// The app never implements transcription itself — it drives the same engine
/// the command line uses, reading `--progress-json` off stderr.
final class Engine {

    /// A running transcription. Pausing is a real SIGSTOP: recording must never
    /// compete with Whisper for the GPU, and a stopped process resumes exactly
    /// where it left off.
    final class Job {
        let folder: URL
        fileprivate let process: Process
        private(set) var isPaused = false

        init(folder: URL, process: Process) {
            self.folder = folder
            self.process = process
        }

        func pause() {
            guard process.isRunning, !isPaused else { return }
            kill(process.processIdentifier, SIGSTOP)
            isPaused = true
        }

        func resume() {
            guard process.isRunning, isPaused else { return }
            kill(process.processIdentifier, SIGCONT)
            isPaused = false
        }

        func cancel() {
            guard process.isRunning else { return }
            resume()   // a stopped process cannot handle a terminate
            process.terminate()
        }
    }

    let cliPath: URL?

    init() {
        cliPath = Engine.locateCLI()
    }

    var isAvailable: Bool { cliPath != nil }

    /// Find the `shumary` CLI without assuming a particular install layout.
    private static func locateCLI() -> URL? {
        if let override = ProcessInfo.processInfo.environment["SHUMARY_CLI"] {
            let url = URL(fileURLWithPath: override)
            if FileManager.default.isExecutableFile(atPath: url.path) { return url }
        }
        if let stored = UserDefaults.standard.string(forKey: "enginePath") {
            let url = URL(fileURLWithPath: stored)
            if FileManager.default.isExecutableFile(atPath: url.path) { return url }
        }

        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let candidates = [
            "\(home)/.local/bin/shumary",
            "/usr/local/bin/shumary",
            "/opt/homebrew/bin/shumary",
            "/Library/Frameworks/Python.framework/Versions/3.11/bin/shumary",
        ]
        for path in candidates where FileManager.default.isExecutableFile(atPath: path) {
            return URL(fileURLWithPath: path)
        }

        // Last resort: ask a login shell, which has the user's real PATH.
        let which = Process()
        which.executableURL = URL(fileURLWithPath: "/bin/zsh")
        which.arguments = ["-lc", "which shumary"]
        let pipe = Pipe()
        which.standardOutput = pipe
        which.standardError = FileHandle.nullDevice
        guard (try? which.run()) != nil else { return nil }
        which.waitUntilExit()
        let output = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if !output.isEmpty, FileManager.default.isExecutableFile(atPath: output) {
            UserDefaults.standard.set(output, forKey: "enginePath")
            return URL(fileURLWithPath: output)
        }
        return nil
    }

    /// Start transcribing `audio`, writing the transcript beside it.
    /// Callbacks arrive on the main thread.
    @discardableResult
    func transcribe(
        folder: URL,
        audio: URL,
        language: String? = nil,
        onProgress: @escaping (Double, String?) -> Void,
        onFinish: @escaping (Result<Void, EngineError>) -> Void
    ) -> Job? {
        guard let cliPath else {
            onFinish(.failure(.notFound))
            return nil
        }

        let process = Process()
        process.executableURL = cliPath
        var args = [
            "transcribe", audio.path,
            "--progress-json",
            "--json",   // transcript.json alongside transcript.md
            "-o", folder.appendingPathComponent("transcript.md").path,
        ]
        if let language = language?.trimmingCharacters(in: .whitespacesAndNewlines), !language.isEmpty {
            args.append(contentsOf: ["--language", language])
        }
        process.arguments = args

        let stderr = Pipe()
        process.standardError = stderr
        process.standardOutput = FileHandle.nullDevice

        var buffer = Data()
        var failure: String?

        stderr.fileHandleForReading.readabilityHandler = { handle in
            buffer.append(handle.availableData)
            // The engine also prints human-readable output; anything that is
            // not a JSON event is simply skipped rather than treated as broken.
            while let newline = buffer.firstIndex(of: 0x0A) {
                let line = buffer.prefix(upTo: newline)
                buffer.removeSubrange(buffer.startIndex...newline)
                guard let event = try? JSONSerialization.jsonObject(with: line) as? [String: Any]
                else { continue }

                if let message = event["message"] as? String, event["event"] as? String == "error" {
                    failure = message
                }
                if let pct = event["pct"] as? Double {
                    let stage = event["stage"] as? String
                    DispatchQueue.main.async { onProgress(pct, stage) }
                }
            }
        }

        process.terminationHandler = { proc in
            stderr.fileHandleForReading.readabilityHandler = nil
            DispatchQueue.main.async {
                if proc.terminationStatus == 0 {
                    onFinish(.success(()))
                } else {
                    onFinish(.failure(.failed(failure ?? "engine exited with code \(proc.terminationStatus)")))
                }
            }
        }

        do {
            try process.run()
        } catch {
            onFinish(.failure(.failed(error.localizedDescription)))
            return nil
        }

        return Job(folder: folder, process: process)
    }

    enum EngineError: Error {
        case notFound
        case failed(String)

        var message: String {
            switch self {
            case .notFound: return "shumary CLI not found"
            case .failed(let why): return why
            }
        }
    }
}
