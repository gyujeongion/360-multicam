# Third-party

| Component | Use | License | Notes |
|---|---|---|---|
| [Light-ASD](https://github.com/Junhua-Liao/Light-ASD) (Liao Junhua, CVPR 2023) | Active speaker detection | MIT | Not vendored. `scripts/setup.sh` clones a pinned commit (`vendor/light_asd.commit`) and applies `vendor/light_asd.patch` (CPU/MPS device, NumPy 2 compatibility). Pretrained weights come from the upstream repository. |
| S3FD face detector weights | Face detection (via Light-ASD) | See upstream | Downloaded by Light-ASD on first use, not redistributed here. |
| [telemetry-parser](https://github.com/AdrianEddy/telemetry-parser) (AdrianEddy) | Reference for the Insta360 gyro record layout and timestamp alignment | MIT OR Apache-2.0 | Layout knowledge only; `multicam360/imu.py` is an independent implementation. |
| [OpenCV Zoo](https://github.com/opencv/opencv_zoo) YuNet face detector, SFace face recognizer | Face detection and identity for tracking | Apache-2.0 (models) | Downloaded to `~/.cache/multicam360/models` on first use, SHA-256 checked, not redistributed. |
| OpenCV, NumPy, SciPy, PyTorch, FFmpeg | Runtime | BSD / Apache-2.0 / LGPL | Installed by the user. |

## Insta360 interoperability

`multicam360/calib.py` reads the factory lens calibration that the camera writes into each `.insv` file (the user's own recording) and implements the published Mei omnidirectional camera model with radial-tangential distortion. No Insta360 software, code or assets are included or redistributed. multicam360 is an independent project, not affiliated with or endorsed by Insta360. Insta360 is a trademark of Arashi Vision Inc.
