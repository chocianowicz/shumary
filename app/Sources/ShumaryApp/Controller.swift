import AppKit
import UniformTypeIdentifiers
import ShumaryRecorder

/// All of the app's behaviour: recording, the transcription queue, and imports.
///
/// The governing rule is that **recording always wins**. A meeting you failed to
/// capture is gone forever; a transcript that arrives twenty minutes late is
/// merely late. So starting a recording pauses whatever the engine is doing, and
/// nothing may ever block or delay a capture.
final class Controller {

    let state = AppState()

    private let recorder = Recorder()
    private let engine = Engine()

    private var currentRecording: URL?
    private var currentJob: Engine.Job?
    private var queue: [URL] = []
    private var pausedJob: Engine.Job?

    /// Set between asking for a recording and the hardware actually starting.
    /// Device setup can block for seconds — the first ever run waits on a TCC
    /// permission dialog — so it happens off the main thread and this guards
    /// against a second ⌘R arriving in the gap.
    private var isStarting = false

    init() {
        state.engineAvailable = engine.isAvailable
    }

    /// Pick up anything a previous run left unfinished.
    func start() {
        queue = SessionStore.outstanding()
        state.queueCount = queue.count
        drainQueue()
    }

    func shutdown() {
        if recorder.isRecording {
            stopRecording()
        }
    }

    // ── Recording ────────────────────────────────────────────────────────────

    func toggleRecording() {
        if recorder.isRecording {
            stopRecording()
        } else if !isStarting {
            startRecording()
        }
    }

    func startRecording() {
        guard !recorder.isRecording, !isStarting else { return }
        isStarting = true
        state.isPreparing = true

        // Settle permission before touching Core Audio. Starting a capture with
        // the question still open blocks forever inside coreaudiod instead of
        // failing, so there would be nothing to report and no way out.
        Permissions.ensureMicrophone { [weak self] granted in
            guard let self else { return }
            guard granted else {
                self.isStarting = false
                self.state.isPreparing = false
                self.state.lastError = Permissions.deniedMessage
                self.state.activity = .idle
                self.resumePausedJob()
                return
            }
            self.beginCapture()
        }
    }

    private func beginCapture() {
        // Recording takes the machine. Whisper can wait.
        if let job = currentJob {
            job.pause()
            pausedJob = job
        }

        let folder: URL
        let audio: URL
        do {
            folder = try Paths.newSessionFolder()
            // Named after the session, not "audio.wav": the engine keys its
            // stage cache on the filename stem, so a shared name would make
            // every session collide with the last one.
            audio = folder.appendingPathComponent("\(folder.lastPathComponent).wav")
        } catch {
            isStarting = false
            state.isPreparing = false
            state.lastError = "could not create a session folder"
            resumePausedJob()
            return
        }

        let metadata = SessionMetadata(
            id: folder.lastPathComponent,
            source: .recorded,
            state: .recording,
            startedAt: ISO8601DateFormatter().string(from: Date()),
            audioFile: audio.lastPathComponent,
            dualSource: true
        )
        SessionStore.save(metadata, to: folder)

        // Off the main thread: creating the tap and aggregate device talks to
        // coreaudiod and, on a first run, waits behind a permission dialog.
        // Doing that on the main thread freezes the menu bar.
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            do {
                try self.recorder.start(writingTo: audio)
                DispatchQueue.main.async {
                    self.isStarting = false
                    self.state.isPreparing = false
                    self.currentRecording = folder
                    self.state.lastError = nil
                    self.state.activity = .recording(startedAt: Date())
                }
            } catch {
                let message = (error as? Recorder.RecorderError)?.description
                    ?? error.localizedDescription
                DispatchQueue.main.async {
                    self.isStarting = false
                    self.state.isPreparing = false
                    // Nothing was captured, so leave no empty session behind.
                    try? FileManager.default.removeItem(at: folder)
                    self.state.lastError = message
                    self.state.activity = .idle
                    self.resumePausedJob()
                    self.drainQueue()
                }
            }
        }
    }

    func stopRecording() {
        guard recorder.isRecording, let folder = currentRecording else { return }

        let duration = recorder.elapsedSeconds
        recorder.stop()
        currentRecording = nil

        writeSidecar(in: folder, audioFile: SessionStore.load(from: folder)?.audioFile ?? "")
        SessionStore.update(folder) { metadata in
            metadata.state = .pending
            metadata.durationSec = duration
        }

        queue.append(folder)
        state.queueCount = queue.count
        state.activity = .idle

        resumePausedJob()
        drainQueue()
    }

    /// Mark the recording as dual-source for the engine, exactly as the CLI does.
    private func writeSidecar(in folder: URL, audioFile: String) {
        guard !audioFile.isEmpty else { return }
        let sidecar = folder.appendingPathComponent("\(audioFile).shumary.json")
        let payload: [String: Any] = [
            "tool": "shumary-record",
            "version": 1,
            "dual_source": true,
            "channels": ["left": "mic", "right": "system"],
            "mic_label": "Me",
            "recorded_at": ISO8601DateFormatter().string(from: Date()),
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: payload, options: .prettyPrinted)
        else { return }
        try? data.write(to: sidecar, options: .atomic)
    }

    // ── Transcription queue ──────────────────────────────────────────────────

    private func resumePausedJob() {
        pausedJob?.resume()
        pausedJob = nil
    }

    private func drainQueue() {
        guard currentJob == nil, !recorder.isRecording, engine.isAvailable else { return }
        guard let folder = queue.first else {
            if case .transcribing = state.activity { state.activity = .idle }
            return
        }
        queue.removeFirst()
        state.queueCount = queue.count

        guard let audio = SessionStore.audioURL(in: folder) else {
            SessionStore.update(folder) { $0.state = .failed; $0.error = "audio file missing" }
            drainQueue()
            return
        }

        SessionStore.update(folder) { $0.state = .transcribing }
        state.activity = .transcribing(session: folder.lastPathComponent, pct: 0)

        currentJob = engine.transcribe(
            folder: folder,
            audio: audio,
            onProgress: { [weak self] pct in
                guard let self, !self.recorder.isRecording else { return }
                self.state.activity = .transcribing(session: folder.lastPathComponent, pct: pct)
            },
            onFinish: { [weak self] result in
                guard let self else { return }
                self.currentJob = nil
                switch result {
                case .success:
                    self.finishSession(folder)
                case .failure(let error):
                    SessionStore.update(folder) {
                        $0.state = .failed
                        $0.error = error.message
                    }
                    self.state.lastError = error.message
                }
                if !self.recorder.isRecording {
                    self.state.activity = .idle
                }
                self.drainQueue()
            }
        )
    }

    /// Compress the raw audio now that a transcript exists.
    private func finishSession(_ folder: URL) {
        guard let metadata = SessionStore.load(from: folder) else { return }
        let audio = folder.appendingPathComponent(metadata.audioFile)

        guard metadata.dualSource, audio.pathExtension == "wav" else {
            SessionStore.update(folder) { $0.state = .done }
            return
        }

        DispatchQueue.global(qos: .utility).async {
            let result = AudioConversion.splitAndCompress(wav: audio, in: folder)
            DispatchQueue.main.async {
                SessionStore.update(folder) { updated in
                    updated.state = .done
                    switch result {
                    case .converted(let mic, let system):
                        updated.tracks = ["mic": mic, "system": system]
                        updated.conversion = "opus"
                    case .skipped(let why):
                        updated.conversion = "skipped: \(why)"
                    }
                }
            }
        }
    }

    // ── Import ───────────────────────────────────────────────────────────────

    func importRecordings() {
        let panel = NSOpenPanel()
        panel.directoryURL = Paths.inbox
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        // Same formats the engine accepts.
        panel.allowedContentTypes = ["m4a", "mp3", "wav", "ogg", "flac", "mp4", "webm", "aac"]
            .compactMap { UTType(filenameExtension: $0) }
        panel.prompt = "Import"
        panel.message = "Choose recordings to transcribe"

        NSApp.activate(ignoringOtherApps: true)
        guard panel.runModal() == .OK else { return }

        for url in panel.urls {
            importOne(url)
        }
        state.queueCount = queue.count
        drainQueue()
    }

    private func importOne(_ url: URL) {
        guard let folder = try? Paths.newSessionFolder() else { return }

        // Session-named for the same cache-collision reason as recordings.
        let destination = folder
            .appendingPathComponent("\(folder.lastPathComponent).\(url.pathExtension)")
        do {
            try FileManager.default.moveItem(at: url, to: destination)
        } catch {
            // Leave anything that failed where the user can still see it.
            try? FileManager.default.removeItem(at: folder)
            state.lastError = "could not import \(url.lastPathComponent)"
            return
        }

        // A file shumary itself recorded carries a sidecar; bring it along so the
        // engine still gets Me/Them instead of falling back to clustering.
        let sidecar = url.appendingPathExtension("shumary.json")
        var dualSource = false
        if FileManager.default.fileExists(atPath: sidecar.path) {
            let moved = destination.appendingPathExtension("shumary.json")
            try? FileManager.default.moveItem(at: sidecar, to: moved)
            dualSource = true
        }

        let metadata = SessionMetadata(
            id: folder.lastPathComponent,
            source: .imported,
            state: .pending,
            startedAt: ISO8601DateFormatter().string(from: Date()),
            audioFile: destination.lastPathComponent,
            dualSource: dualSource,
            originalFilename: url.lastPathComponent
        )
        SessionStore.save(metadata, to: folder)
        queue.append(folder)
    }

    // ── Misc ─────────────────────────────────────────────────────────────────

    func openRecordingsFolder() {
        NSWorkspace.shared.open(Paths.recordings)
    }

    var pendingCount: Int {
        queue.count + (currentJob != nil ? 1 : 0)
    }
}
