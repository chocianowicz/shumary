import Foundation

/// What the app is doing, and the single place the menu reads from.
enum Activity {
    case idle
    case recording(startedAt: Date)
    case transcribing(session: String, pct: Double, stage: String? = nil)
}

/// The verb shown for each engine stage. The percentage alone is not enough:
/// most of the wait is spent in two stages, and saying "transcribing" while the
/// engine is actually separating speakers makes a slow diarization look like a
/// stalled transcription.
private let stageVerbs: [String: String] = [
    "decode": "preparing",
    "transcribe": "transcribing",
    "diarize": "diarizing",
    "align": "aligning",
    "mood": "analyzing",
    "interaction": "analyzing",
    "prosody": "analyzing",
    "render": "writing",
]

final class AppState {
    /// Called on every change so the menu can redraw. Always on the main thread.
    var onChange: (() -> Void)?

    var activity: Activity = .idle { didSet { notify() } }
    var queueCount: Int = 0 { didSet { notify() } }
    var engineAvailable: Bool = true { didSet { notify() } }
    var isPreparing: Bool = false { didSet { notify() } }
    var lastError: String? { didSet { notify() } }

    var isRecording: Bool {
        if case .recording = activity { return true }
        return false
    }

    /// The disabled line at the top of the menu.
    var statusText: String {
        if !engineAvailable, case .idle = activity {
            return "engine not found"
        }

        if isPreparing {
            return "starting…"
        }

        var text: String
        switch activity {
        case .idle:
            text = lastError.map { "failed — \($0)" } ?? "idle"
        case .recording(let startedAt):
            text = "recording · \(Self.clock(Date().timeIntervalSince(startedAt)))"
        case .transcribing(_, let pct, let stage):
            let verb = stage.flatMap { stageVerbs[$0] } ?? "transcribing"
            text = "\(verb) · \(Int(pct * 100))%"
        }

        if queueCount > 0 {
            text += " · \(queueCount) queued"
        }
        return text
    }

    private func notify() {
        if Thread.isMainThread {
            onChange?()
        } else {
            DispatchQueue.main.async { [weak self] in self?.onChange?() }
        }
    }

    static func clock(_ seconds: TimeInterval) -> String {
        let total = Int(max(0, seconds))
        let (h, rem) = total.quotientAndRemainder(dividingBy: 3600)
        let (m, s) = rem.quotientAndRemainder(dividingBy: 60)
        return h > 0 ? String(format: "%d:%02d:%02d", h, m, s)
                     : String(format: "%02d:%02d", m, s)
    }
}
