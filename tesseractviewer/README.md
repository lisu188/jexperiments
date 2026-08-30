# Tesseract Viewer

Native Android/Kotlin viewer for exploring a four-dimensional hypercube.

## Features

- mathematically generated 16-vertex, 32-edge tesseract
- rotations in all six 4D planes: XY, XZ, XW, YZ, YW, ZW
- independent 3D camera orbit
- 4D perspective and orthographic projection
- pinch zoom
- automatic rotation
- dimension-colored edges with W-axis edges emphasized
- no game engine and no runtime dependencies

## Version

Current release: `1.0.0` (`versionCode = 2`).

## Build

The project is intentionally isolated from the repository's JVM Gradle build.

```bash
gradle -p tesseractviewer :app:testDebugUnitTest :app:lintRelease :app:assembleRelease
```

AGP 9.3 requires Gradle 9.5 and JDK 17. GitHub Actions installs those versions explicitly, verifies the APK signature with `apksigner`, generates a SHA-256 checksum, and uploads `TesseractViewer-release-1.0.0`.

The repository does not contain a production signing key. The release variant is minified, resource-shrunk and non-debuggable, but it is signed with Android's debug signing configuration so the generated APK is directly installable for internal use. It must not be treated as a Google Play production artifact or as a stable long-term update channel.

## Controls

- `4D`: drag the model to rotate in the selected 4D plane
- `CAMERA`: drag to orbit the 3D camera around the projected model
- `XY/XZ/XW/YZ/YW/ZW`: select the 4D rotation plane
- pinch: zoom
- `AUTO`: continuously rotate in the selected plane
- `PERSPECTIVE`: switch between 4D perspective and orthographic projection
- `RESET`: restore the default view
