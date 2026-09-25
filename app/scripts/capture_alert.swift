import AppKit
import Foundation
import Darwin

// Use a normal completed AppKit launch so the dialog has a stable app/AX
// identity. The helper exists only while this one fixed-text alert is open.
final class CaptureAlertDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        DispatchQueue.main.async { self.presentAlert() }
    }

    func presentAlert() {
        let app = NSApplication.shared
        let alert = NSAlert()
        alert.alertStyle = .critical
        let mode = CommandLine.arguments.count == 2 ? CommandLine.arguments[1] : "--interrupted"
        if mode == "--startup-battery-critical" {
            alert.messageText = "MeetingIntel recording did not start"
            alert.informativeText = "Battery was too low. Connect power and run mi record again. This attempt saved no audio."
        } else if mode == "--startup-failed" {
            alert.messageText = "MeetingIntel recording did not start"
            alert.informativeText = "Check the failure in Terminal, then run mi record again. Wait for Capture guard active before relying on the recording."
        } else {
            alert.messageText = "MeetingIntel recording interrupted"
            alert.informativeText = "Recording stopped unexpectedly. Later conversation may be missing. Check MeetingIntel for the saved portion before continuing."
        }
        alert.addButton(withTitle: "Understood")
        alert.window.level = .floating
        alert.window.center()
        app.activate(ignoringOtherApps: true)
        var presented = false
        let timer = Timer(timeInterval: 0.1, repeats: true) { _ in
            // Visibility is delivery evidence, never proof someone read it.
            if !presented && alert.window.isVisible {
                presented = true
                print("presented")
                fflush(stdout)
            }
        }
        RunLoop.main.add(timer, forMode: .modalPanel)
        let response = alert.runModal()
        timer.invalidate()
        if response == .alertFirstButtonReturn && presented {
            print("acknowledged")
            fflush(stdout)
            app.terminate(nil)
        } else {
            exit(1)
        }
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = CaptureAlertDelegate()
app.delegate = delegate
// run() calls finishLaunching() before processing the first event. Showing the
// alert from the deferred didFinishLaunching callback preserves that lifecycle.
app.run()
