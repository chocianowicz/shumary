import AppKit

/// shumary — an ultraminimalist menu bar recorder and transcriber.
///
/// No window, no dock icon, no network.
final class AppDelegate: NSObject, NSApplicationDelegate {

    private var controller: Controller!
    private var statusItem: StatusItemController!
    private var hotKey: GlobalHotKey?

    func applicationDidFinishLaunching(_ notification: Notification) {
        Paths.ensure()

        controller = Controller()
        statusItem = StatusItemController(controller: controller)

        // Ask on first launch, so the permission dialog is answered long before
        // someone is mid-meeting reaching for the record shortcut.
        Permissions.primeAtLaunch()

        // Works from anywhere, including mid-call with another app frontmost.
        hotKey = GlobalHotKey { [weak self] in
            self?.controller.toggleRecording()
        }

        controller.start()
    }

    func applicationWillTerminate(_ notification: Notification) {
        // Never drop a capture on the way out.
        controller.shutdown()
    }
}

let application = NSApplication.shared
let delegate = AppDelegate()
application.delegate = delegate
application.setActivationPolicy(.accessory)   // menu bar only
application.run()
