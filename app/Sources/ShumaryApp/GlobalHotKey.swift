import AppKit
import Carbon.HIToolbox

/// A system-wide hotkey, registered through Carbon deliberately.
///
/// A menu bar app never becomes frontmost, so an `NSMenuItem` key equivalent
/// only fires while its menu is already open — useless mid-call, when Zoom owns
/// the keyboard. The modern alternative (`NSEvent.addGlobalMonitorForEvents`)
/// needs Accessibility permission; `RegisterEventHotKey` needs none.
final class GlobalHotKey {

    private var hotKeyRef: EventHotKeyRef?
    private var handlerRef: EventHandlerRef?
    private let action: () -> Void

    /// `⌃⌥⌘R` — deliberately not plain ⌘R, which is reload in every browser and
    /// editor on the machine and would break all of them if claimed globally.
    init?(action: @escaping () -> Void) {
        self.action = action

        var eventType = EventTypeSpec(
            eventClass: OSType(kEventClassKeyboard),
            eventKind: UInt32(kEventHotKeyPressed)
        )

        let callback: EventHandlerUPP = { _, event, userData in
            guard let userData, let event else { return noErr }
            var hotKeyID = EventHotKeyID()
            GetEventParameter(event, EventParamName(kEventParamDirectObject),
                              EventParamType(typeEventHotKeyID), nil,
                              MemoryLayout<EventHotKeyID>.size, nil, &hotKeyID)
            let instance = Unmanaged<GlobalHotKey>.fromOpaque(userData).takeUnretainedValue()
            DispatchQueue.main.async { instance.action() }
            return noErr
        }

        let selfPtr = Unmanaged.passUnretained(self).toOpaque()
        guard InstallEventHandler(GetApplicationEventTarget(), callback, 1,
                                  &eventType, selfPtr, &handlerRef) == noErr else {
            return nil
        }

        let hotKeyID = EventHotKeyID(signature: OSType(0x53_4D_52_43), id: 1)  // 'SMRC'
        let modifiers = UInt32(controlKey | optionKey | cmdKey)
        guard RegisterEventHotKey(UInt32(kVK_ANSI_R), modifiers, hotKeyID,
                                  GetApplicationEventTarget(), 0, &hotKeyRef) == noErr else {
            return nil
        }
    }

    deinit {
        if let hotKeyRef { UnregisterEventHotKey(hotKeyRef) }
        if let handlerRef { RemoveEventHandler(handlerRef) }
    }
}
