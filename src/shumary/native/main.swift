// shumary-record — CLI front end for `Recorder`.
//
// Build:  swiftc -O Recorder.swift main.swift -o shumary-record \
//             -framework CoreAudio -framework AudioToolbox \
//             -framework AVFoundation -framework Foundation
//
// Usage:  shumary-record <output.wav>       (stop with SIGINT / SIGTERM)
//
// Everything that is specific to being a command line process lives here:
// argument parsing, stderr logging, signal handling, the status line, and the
// parent-death watchdog. The capture itself is in Recorder.swift, shared with
// the menu bar app.

import Foundation

func note(_ msg: String) {
    FileHandle.standardError.write(("  " + msg + "\n").data(using: .utf8)!)
}

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(("shumary-record: " + msg + "\n").data(using: .utf8)!)
    exit(1)
}

let args = CommandLine.arguments
guard args.count >= 2 else { fail("usage: shumary-record <output.wav>") }
let outputPath = args[1]

guard #available(macOS 14.4, *) else {
    fail("system audio capture needs macOS 14.4 or newer (process taps).")
}

let recorder = Recorder()
recorder.onLog = { note($0) }

// Teardown is idempotent inside `Recorder`, so SIGINT racing the parent watch
// is harmless.
func stopAndExit() {
    recorder.stop()
    note("written to \(outputPath)")
    exit(0)
}

for sig in [SIGINT, SIGTERM] {
    signal(sig, SIG_IGN)
    let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    source.setEventHandler { stopAndExit() }
    source.resume()
    // Keep the source alive for the process lifetime.
    _ = Unmanaged.passRetained(source as AnyObject)
}

// Stop if whoever launched us goes away. Without this an interrupted `shumary
// record` would leave the helper recording forever, holding a system-audio tap
// nobody is watching. This is CLI-only — in the app the recorder is in-process
// and there is no parent to outlive.
let parentPID = getppid()
if parentPID > 1 {
    let parentWatch = DispatchSource.makeProcessSource(identifier: parentPID, eventMask: .exit, queue: .main)
    parentWatch.setEventHandler {
        note("parent process exited")
        stopAndExit()
    }
    parentWatch.resume()
    _ = Unmanaged.passRetained(parentWatch as AnyObject)
}

do {
    try recorder.start(writingTo: URL(fileURLWithPath: outputPath))
} catch let error as Recorder.RecorderError {
    fail(error.description)
} catch {
    fail(error.localizedDescription)
}

note("recording — press Ctrl-C to stop")

// Status line: elapsed time plus peak level meters for both channels.
let ticker = DispatchSource.makeTimerSource(queue: .main)
ticker.schedule(deadline: .now() + 0.5, repeating: 0.5)
ticker.setEventHandler {
    let seconds = recorder.elapsedSeconds
    let levels = recorder.currentLevels()
    func meter(_ level: Float) -> String {
        let filled = Int((min(1, level) * 12).rounded())
        return String(repeating: "█", count: filled) + String(repeating: "·", count: 12 - filled)
    }
    let line = String(format: "\r  %02d:%02d  you %@  system %@",
                      Int(seconds) / 60, Int(seconds) % 60,
                      meter(levels.mic), meter(levels.system))
    FileHandle.standardError.write(line.data(using: .utf8)!)
}
ticker.resume()

RunLoop.main.run()
