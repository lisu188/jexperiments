# Tesseract Viewer

Native Android/Kotlin viewer for exploring a four-dimensional hypercube.

## Features

- mathematically generated 16-vertex, 32-edge tesseract
- rotations in all six 4D planes: XY, XZ, XW, YZ, YW, ZW
- independent 3D camera orbit
- 4D perspective and orthographic projection
- pinch zoom
- automatic rotation
- dimension-colored edges with the active rotation dimensions emphasized
- adaptive portrait/landscape controls with Android system-bar inset handling
- accessible 48 dp controls implemented as native Android views
- no game engine and no runtime dependencies

## Version

Current development release: `1.1.0` (`versionCode = 3`).

## Build

The project is intentionally isolated from the repository's JVM Gradle build.

```bash
gradle -p tesseractviewer :app:testDebugUnitTest :app:lintRelease :app:assembleRelease
```

AGP 9.3 requires Gradle 9.5 and JDK 17. GitHub Actions installs those versions explicitly, verifies the APK signature with `apksigner`, generates a SHA-256 checksum, and uploads `TesseractViewer-release-1.1.0`.

The repository does not contain a production signing key. The release variant is minified, resource-shrunk and non-debuggable, but it is signed with Android's debug signing configuration so the generated APK is directly installable for internal use. It must not be treated as a Google Play production artifact or as a stable long-term update channel.

## Controls

- `4D rotation`: drag the model to rotate in the selected plane
- `3D camera`: drag to orbit the camera around the already-projected 3D model
- `XW`, `YW`, `ZW`: fourth-dimensional planes, surfaced first in the UI
- `XY`, `XZ`, `YZ`: familiar spatial rotation planes
- pinch: zoom in either interaction mode
- `Auto rotate`: continuously rotate in the selected 4D plane
- `Perspective` / `Orthographic`: switch the 4D projection method
- `Reset`: restore the default XW view

The colored X/Y/Z/W legend matches the edge colors. In 4D mode, edges belonging to the selected rotation plane are emphasized while unrelated dimensions are visually reduced.
