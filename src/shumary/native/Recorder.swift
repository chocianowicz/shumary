// Recorder — capture microphone + system audio into one stereo WAV.
//
//   channel L = microphone (you)
//   channel R = system output (everyone else)
//
// Uses Core Audio process taps (macOS 14.4+): a global output tap and the
// default input device are combined into a single private aggregate device, so
// both sources share one clock and cannot drift apart over a long recording.
// No virtual audio driver, no admin rights, and system output keeps playing
// through the speakers while it is tapped.
//
// Shared by the `shumary-record` CLI helper (see main.swift) and the menu bar
// app. Nothing in this file may call `exit()` or write to stderr — an app must
// survive a missing device, and its logging is its own business (`onLog`).

import Foundation
import CoreAudio
import AudioToolbox
import AVFoundation

// ── Core Audio property helpers ──────────────────────────────────────────────

/// Four-char-code rendering for OSStatus, which is how Core Audio reports errors.
func shumaryStatusText(_ s: OSStatus) -> String {
    let n = UInt32(bitPattern: s)
    let chars = [UInt8((n >> 24) & 255), UInt8((n >> 16) & 255),
                 UInt8((n >> 8) & 255), UInt8(n & 255)]
    if chars.allSatisfy({ $0 >= 32 && $0 < 127 }),
       let ascii = String(bytes: chars, encoding: .ascii) {
        return "'\(ascii)' (\(s))"
    }
    return "\(s)"
}

private func address(_ selector: AudioObjectPropertySelector,
                     _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal)
    -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: selector,
                               mScope: scope,
                               mElement: kAudioObjectPropertyElementMain)
}

private func objectID(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> AudioObjectID? {
    var addr = address(selector)
    var value = AudioObjectID(kAudioObjectUnknown)
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    let status = AudioObjectGetPropertyData(object, &addr, 0, nil, &size, &value)
    return status == noErr && value != kAudioObjectUnknown ? value : nil
}

private func stringProperty(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String? {
    var addr = address(selector)
    var value: CFString? = nil
    var size = UInt32(MemoryLayout<CFString?>.size)
    let status = withUnsafeMutablePointer(to: &value) {
        AudioObjectGetPropertyData(object, &addr, 0, nil, &size, $0)
    }
    guard status == noErr else { return nil }
    return value as String?
}

private func nominalSampleRate(_ device: AudioObjectID) throws -> Double {
    var addr = address(kAudioDevicePropertyNominalSampleRate)
    var rate: Double = 0
    var size = UInt32(MemoryLayout<Double>.size)
    let status = AudioObjectGetPropertyData(device, &addr, 0, nil, &size, &rate)
    guard status == noErr else { throw Recorder.RecorderError.coreAudio("read sample rate", status) }
    return rate
}

/// Channel count of each input buffer the device will hand to an IOProc, in order.
private func inputBufferChannelCounts(_ device: AudioObjectID) throws -> [Int] {
    var addr = address(kAudioDevicePropertyStreamConfiguration, kAudioObjectPropertyScopeInput)
    var size: UInt32 = 0
    var status = AudioObjectGetPropertyDataSize(device, &addr, 0, nil, &size)
    guard status == noErr else { throw Recorder.RecorderError.coreAudio("read stream config size", status) }

    let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(size),
                                               alignment: MemoryLayout<AudioBufferList>.alignment)
    defer { raw.deallocate() }
    status = AudioObjectGetPropertyData(device, &addr, 0, nil, &size, raw)
    guard status == noErr else { throw Recorder.RecorderError.coreAudio("read stream config", status) }

    let listPtr = raw.assumingMemoryBound(to: AudioBufferList.self)
    return UnsafeMutableAudioBufferListPointer(listPtr).map { Int($0.mNumberChannels) }
}

// ── WAV writer ───────────────────────────────────────────────────────────────

/// Streaming 16-bit stereo WAV writer. The RIFF/data sizes are unknown until the
/// recording stops, so a placeholder header is written up front and patched on close.
final class WavWriter {
    private let handle: FileHandle
    private let rate: Int
    private var frames: Int = 0
    private let lock = NSLock()

    private static let headerBytes = 44

    init(path: String, sampleRate: Int) throws {
        FileManager.default.createFile(atPath: path, contents: nil)
        guard let h = FileHandle(forWritingAtPath: path) else {
            throw Recorder.RecorderError.cannotWrite(path)
        }
        self.handle = h
        self.rate = sampleRate
        try h.write(contentsOf: Data(count: Self.headerBytes))  // placeholder
    }

    /// Append interleaved stereo frames (L, R, L, R, …) as 16-bit PCM.
    func append(_ interleaved: [Int16]) {
        lock.lock()
        defer { lock.unlock() }
        interleaved.withUnsafeBufferPointer { buf in
            try? handle.write(contentsOf: Data(buffer: buf))
        }
        frames += interleaved.count / 2
    }

    var frameCount: Int {
        lock.lock(); defer { lock.unlock() }
        return frames
    }

    private func header(forFrames frameCount: Int) -> Data {
        let channels = 2, bitsPerSample = 16
        let bytesPerFrame = channels * bitsPerSample / 8
        let dataBytes = frameCount * bytesPerFrame

        var header = Data()
        func ascii(_ s: String) { header.append(contentsOf: Array(s.utf8)) }
        func u32(_ v: Int) { var le = UInt32(truncatingIfNeeded: v).littleEndian; withUnsafeBytes(of: &le) { header.append(contentsOf: $0) } }
        func u16(_ v: Int) { var le = UInt16(truncatingIfNeeded: v).littleEndian; withUnsafeBytes(of: &le) { header.append(contentsOf: $0) } }

        ascii("RIFF"); u32(36 + dataBytes); ascii("WAVE")
        ascii("fmt "); u32(16); u16(1); u16(channels)
        u32(rate); u32(rate * bytesPerFrame); u16(bytesPerFrame); u16(bitsPerSample)
        ascii("data"); u32(dataBytes)
        return header
    }

    /// Rewrite the header in place, then seek back to the end to keep appending.
    ///
    /// Called periodically so a recording that dies without running `close()`
    /// (a crash, SIGKILL, a laptop shutting down mid-meeting) is still a valid
    /// WAV up to the last checkpoint, rather than a file with a zero-length
    /// data chunk that nothing will open.
    func checkpoint() {
        lock.lock()
        defer { lock.unlock() }
        guard let end = try? handle.offset() else { return }
        try? handle.seek(toOffset: 0)
        try? handle.write(contentsOf: header(forFrames: frames))
        try? handle.seek(toOffset: end)
    }

    func close() {
        lock.lock()
        defer { lock.unlock() }
        try? handle.seek(toOffset: 0)
        try? handle.write(contentsOf: header(forFrames: frames))
        try? handle.close()
    }
}

// ── Channel routing ──────────────────────────────────────────────────────────

/// Where the mic and tap audio live inside the aggregate device's input buffers.
///
/// The aggregate presents one buffer per member stream, sub-devices first and
/// taps after, so the mic is the first buffer and the tap the last. When Core
/// Audio collapses everything into a single interleaved buffer instead, both
/// sources share buffer 0 and are told apart by channel offset.
struct Routing {
    let micBuffer: Int
    let micChannel: Int
    let tapBuffer: Int
    let tapChannel: Int
    let tapChannels: Int

    static func resolve(counts: [Int], micChannels: Int, tapChannels: Int) -> Routing {
        if counts.count >= 2 {
            return Routing(micBuffer: 0, micChannel: 0,
                           tapBuffer: counts.count - 1, tapChannel: 0,
                           tapChannels: counts[counts.count - 1])
        }
        // Single fused buffer: mic channels first, then the tap's.
        let total = counts.first ?? 0
        return Routing(micBuffer: 0, micChannel: 0,
                       tapBuffer: 0, tapChannel: micChannels,
                       tapChannels: max(1, min(tapChannels, total - micChannels)))
    }
}

// ── Recorder ─────────────────────────────────────────────────────────────────

@available(macOS 14.4, *)
public final class Recorder {

    public enum RecorderError: Error, CustomStringConvertible {
        case alreadyRecording
        case noInputDevice
        case noDeviceUID
        case tapDenied(OSStatus)
        case cannotWrite(String)
        case coreAudio(String, OSStatus)

        public var description: String {
            switch self {
            case .alreadyRecording:
                return "a recording is already in progress"
            case .noInputDevice:
                return "no default input device — is a microphone connected?"
            case .noDeviceUID:
                return "could not read the microphone's device UID"
            case .tapDenied(let status):
                return """
                could not create the system-audio tap — \(shumaryStatusText(status)). \
                macOS requires permission for this: grant audio recording access under \
                System Settings → Privacy & Security, then retry.
                """
            case .cannotWrite(let path):
                return "cannot open \(path) for writing"
            case .coreAudio(let what, let status):
                return "\(what) failed — \(shumaryStatusText(status))"
            }
        }
    }

    public struct Levels {
        public let mic: Float
        public let system: Float
    }

    /// Diagnostic messages. The CLI sends these to stderr; the app can ignore them.
    public var onLog: ((String) -> Void)?

    public private(set) var isRecording = false
    public private(set) var sampleRate: Double = 0

    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggregateID = AudioObjectID(kAudioObjectUnknown)
    private var procID: AudioDeviceIOProcID?
    private var writer: WavWriter?
    private var checkpointTimer: DispatchSourceTimer?

    // Peak levels, written by the realtime IOProc and read by whoever is
    // drawing a meter. A plain Float pair is enough — a stale read costs one
    // frame of meter, and taking a lock on the audio thread would not.
    private let levels = UnsafeMutablePointer<Float>.allocate(capacity: 2)

    private let silenceEngine = AVAudioEngine()
    private let silencePlayer = AVAudioPlayerNode()

    private let lock = NSLock()

    public init() {
        levels[0] = 0
        levels[1] = 0
    }

    deinit {
        levels.deallocate()
    }

    public var frameCount: Int { writer?.frameCount ?? 0 }

    public var elapsedSeconds: Double {
        sampleRate > 0 ? Double(frameCount) / sampleRate : 0
    }

    public func currentLevels() -> Levels {
        Levels(mic: levels[0], system: levels[1])
    }

    // ── Start ────────────────────────────────────────────────────────────────

    public func start(writingTo url: URL) throws {
        lock.lock()
        if isRecording {
            lock.unlock()
            throw RecorderError.alreadyRecording
        }
        lock.unlock()

        // 1. Microphone (default input device).
        guard let micDevice = objectID(AudioObjectID(kAudioObjectSystemObject),
                                       kAudioHardwarePropertyDefaultInputDevice) else {
            throw RecorderError.noInputDevice
        }
        guard let micUID = stringProperty(micDevice, kAudioDevicePropertyDeviceUID) else {
            throw RecorderError.noDeviceUID
        }
        let micName = stringProperty(micDevice, kAudioObjectPropertyName) ?? "microphone"
        let micChannelCount = try inputBufferChannelCounts(micDevice).reduce(0, +)
        onLog?("mic: \(micName) (\(micChannelCount) ch)")

        // 2. Global output tap, left unmuted so playback still reaches the speakers.
        let tapDescription = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
        tapDescription.name = "Shumary System Audio"
        tapDescription.uuid = UUID()
        tapDescription.muteBehavior = .unmuted
        tapDescription.isPrivate = true

        let tapStatus = AudioHardwareCreateProcessTap(tapDescription, &tapID)
        guard tapStatus == noErr else { throw RecorderError.tapDenied(tapStatus) }
        onLog?("system tap: created")

        // 3. One private aggregate device carrying both, so they share a clock.
        let aggregateDescription: [String: Any] = [
            kAudioAggregateDeviceNameKey: "Shumary Recorder",
            kAudioAggregateDeviceUIDKey: "com.shumary.recorder.\(UUID().uuidString)",
            kAudioAggregateDeviceMainSubDeviceKey: micUID,   // mic is the clock master
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: micUID]],
            kAudioAggregateDeviceTapListKey: [[
                kAudioSubTapDriftCompensationKey: true,
                kAudioSubTapUIDKey: tapDescription.uuid.uuidString,
            ]],
        ]

        let aggregateStatus = AudioHardwareCreateAggregateDevice(
            aggregateDescription as CFDictionary, &aggregateID)
        guard aggregateStatus == noErr else {
            cleanUpAfterFailedStart()
            throw RecorderError.coreAudio("create aggregate device", aggregateStatus)
        }

        let rate = try nominalSampleRate(aggregateID)
        let bufferCounts = try inputBufferChannelCounts(aggregateID)
        let routing = Routing.resolve(counts: bufferCounts,
                                      micChannels: micChannelCount,
                                      tapChannels: 2)
        onLog?("aggregate: \(Int(rate)) Hz, input buffers \(bufferCounts)")

        let writer: WavWriter
        do {
            writer = try WavWriter(path: url.path, sampleRate: Int(rate))
        } catch {
            cleanUpAfterFailedStart()
            throw error
        }

        // 4. Pull both sources every cycle and interleave them into L=mic, R=system.
        //    The block deliberately captures the writer, routing and level buffer
        //    rather than `self`, so the audio thread never touches the class.
        let levelsPtr = levels
        let ioStatus = AudioDeviceCreateIOProcIDWithBlock(&procID, aggregateID, nil) {
            _, inInputData, _, _, _ in
            let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inInputData))
            guard buffers.count > max(routing.micBuffer, routing.tapBuffer) else { return }

            let micBuf = buffers[routing.micBuffer]
            let tapBuf = buffers[routing.tapBuffer]
            guard let micRaw = micBuf.mData, let tapRaw = tapBuf.mData else { return }

            let micStride = Int(micBuf.mNumberChannels)
            let tapStride = Int(tapBuf.mNumberChannels)
            guard micStride > 0, tapStride > 0 else { return }

            let mic = micRaw.assumingMemoryBound(to: Float.self)
            let tap = tapRaw.assumingMemoryBound(to: Float.self)
            let micFrames = Int(micBuf.mDataByteSize) / (MemoryLayout<Float>.size * micStride)
            let tapFrames = Int(tapBuf.mDataByteSize) / (MemoryLayout<Float>.size * tapStride)
            let frames = min(micFrames, tapFrames)
            guard frames > 0 else { return }

            // How many tap channels to fold down into the single system channel.
            let tapFold = max(1, min(routing.tapChannels, tapStride - routing.tapChannel))

            var out = [Int16](repeating: 0, count: frames * 2)
            var micPeak: Float = 0
            var tapPeak: Float = 0

            for frame in 0..<frames {
                let micSample = mic[frame * micStride + routing.micChannel]

                var tapSample: Float = 0
                for ch in 0..<tapFold {
                    tapSample += tap[frame * tapStride + routing.tapChannel + ch]
                }
                tapSample /= Float(tapFold)

                micPeak = max(micPeak, abs(micSample))
                tapPeak = max(tapPeak, abs(tapSample))

                out[frame * 2]     = Int16(max(-1, min(1, micSample)) * 32767)
                out[frame * 2 + 1] = Int16(max(-1, min(1, tapSample)) * 32767)
            }

            levelsPtr[0] = micPeak
            levelsPtr[1] = tapPeak
            writer.append(out)
        }
        guard ioStatus == noErr else {
            writer.close()
            cleanUpAfterFailedStart()
            throw RecorderError.coreAudio("create IO proc", ioStatus)
        }

        self.writer = writer
        self.sampleRate = rate

        startSilence()

        let startStatus = AudioDeviceStart(aggregateID, procID)
        guard startStatus == noErr else {
            silencePlayer.stop()
            silenceEngine.stop()
            writer.close()
            cleanUpAfterFailedStart()
            self.writer = nil
            throw RecorderError.coreAudio("start recording", startStatus)
        }

        startCheckpointTimer()

        lock.lock()
        isRecording = true
        lock.unlock()
    }

    // ── Stop ─────────────────────────────────────────────────────────────────

    /// Tear down and finalise the WAV. Idempotent, and safe from any thread.
    public func stop() {
        lock.lock()
        guard isRecording else { lock.unlock(); return }
        isRecording = false
        lock.unlock()

        checkpointTimer?.cancel()
        checkpointTimer = nil

        silencePlayer.stop()
        silenceEngine.stop()

        if let procID {
            AudioDeviceStop(aggregateID, procID)
            AudioDeviceDestroyIOProcID(aggregateID, procID)
        }
        procID = nil

        if aggregateID != kAudioObjectUnknown {
            AudioHardwareDestroyAggregateDevice(aggregateID)
            aggregateID = AudioObjectID(kAudioObjectUnknown)
        }
        if tapID != kAudioObjectUnknown {
            AudioHardwareDestroyProcessTap(tapID)
            tapID = AudioObjectID(kAudioObjectUnknown)
        }

        let seconds = elapsedSeconds
        writer?.close()
        onLog?(String(format: "stopped — %.1fs captured", seconds))
    }

    /// Release whatever was built before a failure part-way through `start`.
    private func cleanUpAfterFailedStart() {
        if aggregateID != kAudioObjectUnknown {
            AudioHardwareDestroyAggregateDevice(aggregateID)
            aggregateID = AudioObjectID(kAudioObjectUnknown)
        }
        if tapID != kAudioObjectUnknown {
            AudioHardwareDestroyProcessTap(tapID)
            tapID = AudioObjectID(kAudioObjectUnknown)
        }
    }

    // ── Keeping the output device awake ──────────────────────────────────────

    /// A process tap only produces frames while the output device is actually
    /// running, and an aggregate device delivers nothing until every member
    /// stream does — so on a silent machine the IOProc never fires and even the
    /// mic is lost. Playing inaudible silence for the duration keeps the output
    /// device running, so recording works whether or not a call has started yet.
    private func startSilence() {
        let format = silenceEngine.outputNode.outputFormat(forBus: 0)
        guard format.sampleRate > 0,
              let buffer = AVAudioPCMBuffer(pcmFormat: format,
                                            frameCapacity: AVAudioFrameCount(format.sampleRate))
        else {
            onLog?("warning: could not hold the output device open — recording may not "
                   + "capture anything until audio starts playing")
            return
        }
        buffer.frameLength = buffer.frameCapacity  // zero-filled == silence
        silenceEngine.attach(silencePlayer)
        silenceEngine.connect(silencePlayer, to: silenceEngine.mainMixerNode, format: format)
        do {
            try silenceEngine.start()
            silencePlayer.scheduleBuffer(buffer, at: nil, options: .loops)
            silencePlayer.play()
        } catch {
            onLog?("warning: could not hold the output device open — \(error.localizedDescription)")
        }
    }

    /// Patch the WAV header every 10s so an unclean exit still leaves a
    /// playable file.
    private func startCheckpointTimer() {
        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "shumary.checkpoint"))
        timer.schedule(deadline: .now() + 10, repeating: 10)
        timer.setEventHandler { [weak self] in self?.writer?.checkpoint() }
        timer.resume()
        checkpointTimer = timer
    }
}
