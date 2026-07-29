// swift-tools-version:5.9
//
// The menu bar app. Lives at the repo root rather than under app/ because the
// recorder it depends on is shared verbatim with the Python CLI helper in
// src/shumary/native/ — SwiftPM requires every target to live inside the
// package root, so the package root has to contain both.
//
//   swift build -c release          # binary only
//   ./app/scripts/build-app.sh      # assembled, signed shumary.app

import PackageDescription

let package = Package(
    name: "Shumary",
    platforms: [.macOS("14.4")],   // Core Audio process taps
    targets: [
        // Shared with `shumary-record` (src/shumary/native/main.swift), which
        // the Python package compiles on demand. Excluded here because a target
        // may not contain top-level statements.
        .target(
            name: "ShumaryRecorder",
            path: "src/shumary/native",
            exclude: ["main.swift", "__init__.py"]
        ),
        .executableTarget(
            name: "ShumaryApp",
            dependencies: ["ShumaryRecorder"],
            path: "app/Sources/ShumaryApp"
        ),
    ]
)
