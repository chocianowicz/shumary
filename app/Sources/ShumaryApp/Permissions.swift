import AVFoundation

/// Microphone authorisation, asked for explicitly.
///
/// This exists because of how Core Audio behaves when permission has not been
/// decided yet: `AudioDeviceCreateIOProcIDWithBlock` blocks in `mach_msg`
/// waiting on coreaudiod *indefinitely* rather than returning an error, so a
/// recording started before the user has answered simply hangs with no way to
/// report why.
///
/// `AVCaptureDevice.requestAccess` raises the same prompt without blocking, and
/// gives a definite yes or no to act on.
enum Permissions {

    enum Status {
        case granted
        case denied
        case undetermined
    }

    static var microphone: Status {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: return .granted
        case .notDetermined: return .undetermined
        default: return .denied
        }
    }

    /// Ask once, on first launch, so the question is settled long before a
    /// meeting starts rather than in the moment someone hits record.
    static func primeAtLaunch() {
        guard microphone == .undetermined else { return }
        AVCaptureDevice.requestAccess(for: .audio) { _ in }
    }

    /// Resolve permission, prompting if it has never been asked.
    /// The completion always lands on the main thread.
    static func ensureMicrophone(_ completion: @escaping (Bool) -> Void) {
        switch microphone {
        case .granted:
            completion(true)
        case .denied:
            completion(false)
        case .undetermined:
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                DispatchQueue.main.async { completion(granted) }
            }
        }
    }

    static let deniedMessage =
        "microphone access denied — enable shumary under System Settings → Privacy & Security → Microphone"
}
