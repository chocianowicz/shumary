#!/usr/bin/env bash
# Assemble dist/shumary.app from the SwiftPM build.
#
# Process taps require a signed binary, so the bundle is ad-hoc signed. If the
# tap works from the CLI but returns silence inside the app, signing is the
# first thing to check.
set -euo pipefail

cd "$(dirname "$0")/../.."

swift build -c release --product ShumaryApp

APP="dist/shumary.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp app/Info.plist "$APP/Contents/Info.plist"
cp app/Resources/*.png "$APP/Contents/Resources/"
cp "$(swift build -c release --product ShumaryApp --show-bin-path)/ShumaryApp" \
   "$APP/Contents/MacOS/shumary"

codesign --force --sign - --timestamp=none "$APP"

echo "built $APP"
