import AppKit

/// The entire user interface: two letters in the menu bar and four actions.
final class StatusItemController: NSObject, NSMenuDelegate {

    private let statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
    private let controller: Controller
    private let menu = NSMenu()

    private let statusLine = NSMenuItem(title: "idle", action: nil, keyEquivalent: "")
    private let recordItem = NSMenuItem(title: "Start recording", action: #selector(toggleRecording), keyEquivalent: "r")
    private let importItem = NSMenuItem(title: "Import recording…", action: #selector(importRecordings), keyEquivalent: "u")
    private let openItem = NSMenuItem(title: "Open recordings folder", action: #selector(openFolder), keyEquivalent: "o")
    private let quitItem = NSMenuItem(title: "Quit shumary", action: #selector(quit), keyEquivalent: "q")

    /// Refreshes the elapsed time while the menu is open.
    private var menuTimer: Timer?

    init(controller: Controller) {
        self.controller = controller
        super.init()

        statusLine.isEnabled = false

        for item in [recordItem, importItem, openItem, quitItem] {
            item.target = self
        }

        menu.addItem(statusLine)
        menu.addItem(.separator())
        menu.addItem(recordItem)
        menu.addItem(importItem)
        menu.addItem(openItem)
        menu.addItem(.separator())
        menu.addItem(quitItem)
        menu.delegate = self

        statusItem.menu = menu

        controller.state.onChange = { [weak self] in self?.refresh() }
        refresh()
    }

    // ── Appearance ───────────────────────────────────────────────────────────

    /// The shumary mark, plus a steady red dot while recording.
    ///
    /// The mark is a *template* image: macOS ignores its colour and re-tints it
    /// from the alpha channel, so it follows the menu bar in light and dark mode
    /// on its own. The dot is drawn as title text instead, because it must stay
    /// red in both — a template image cannot keep a colour.
    ///
    /// The dot never blinks. A pulsing menu bar item reads as an alert demanding
    /// attention; this is a state indicator.
    private lazy var markImage: NSImage? = {
        guard let image = NSImage(named: "MenuBarIcon") else { return nil }
        image.isTemplate = true
        image.size = NSSize(width: image.size.width * Self.markHeight / image.size.height,
                            height: Self.markHeight)
        return image
    }()

    private static let markHeight: CGFloat = 18

    private func updateIcon(recording: Bool) {
        guard let button = statusItem.button else { return }

        let dot = recording
            ? NSAttributedString(
                string: " ●",
                attributes: [
                    .foregroundColor: NSColor.systemRed,
                    .font: NSFont.systemFont(ofSize: 9),
                ]
              )
            : NSAttributedString(string: "")

        if let markImage {
            button.image = markImage
            button.imagePosition = recording ? .imageLeading : .imageOnly
            button.attributedTitle = dot
        } else {
            // Resources missing: fall back to the wordmark rather than leaving
            // an invisible menu bar item.
            button.image = nil
            let title = NSMutableAttributedString(
                string: "SM",
                attributes: [
                    .foregroundColor: NSColor.labelColor,
                    .font: NSFont.systemFont(ofSize: 12, weight: .semibold),
                ]
            )
            title.append(dot)
            button.attributedTitle = title
        }
    }

    private func refresh() {
        let state = controller.state
        updateIcon(recording: state.isRecording)
        statusLine.title = state.statusText
        recordItem.title = state.isRecording ? "Stop recording" : "Start recording"
        importItem.isEnabled = true

        let pending = controller.pendingCount
        quitItem.title = pending > 0 ? "Quit shumary (\(pending) pending)" : "Quit shumary"
    }

    // ── Menu lifetime ────────────────────────────────────────────────────────

    func menuWillOpen(_ menu: NSMenu) {
        refresh()
        // Only tick while the user is actually looking at the menu.
        menuTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            self?.statusLine.title = self?.controller.state.statusText ?? ""
        }
        if let menuTimer {
            RunLoop.main.add(menuTimer, forMode: .common)
        }
    }

    func menuDidClose(_ menu: NSMenu) {
        menuTimer?.invalidate()
        menuTimer = nil
    }

    // ── Actions ──────────────────────────────────────────────────────────────

    @objc func toggleRecording() { controller.toggleRecording() }
    @objc func importRecordings() { controller.importRecordings() }
    @objc func openFolder() { controller.openRecordingsFolder() }
    @objc func quit() { NSApp.terminate(nil) }
}
