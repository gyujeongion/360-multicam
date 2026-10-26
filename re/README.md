# Calibration format notes

How `orbcut/calib.py` gets from a `.insv` recording to per-lens projection parameters. Each point was checked
against the camera's own files and by rendering.

## Where the calibration lives

An `.insv` file ends with a trailer:

- last 32 bytes: ASCII magic `8db42d694ccc418790edff439fe026bf`
- before it: `uint32` version (3) and `uint32` trailer length
- records are read back to front, each laid out as `[data][uint16 id][uint32 size]`

Record `0x101` is a protobuf metadata message (about 1.5 KB). One of its string fields holds the lens
calibration. The X6 writes the `offset_v6` variant; the gyro records used by `orbcut/imu.py` sit in the same
trailer (layout as documented by [telemetry-parser](https://github.com/AdrianEddy/telemetry-parser)).

## `offset_v6`

Two lens blocks of 27 values each, plus one trailing value whose meaning is unverified:

```
xi  fx  fy  cx  cy | rot(3, degrees) | trans(3) | dist(13) | W  H  fov(full angle, degrees)
```

- Coordinates refer to both lenses side by side on a 15488 × 7744 canvas (lens 1's `cx` is offset by 7744).
  Stream pixels = (calibration pixels − 32) × 0.5: a centred 7744 → 7680 crop, then the 3840 stream.
- `fov` is the full angle (193°). The model below expects the half angle; using the full angle makes the
  lens overlap jump to 84 %, an easy way to spot the mistake.
- Rotation: lens 1 is `R1ᵀ·R0` (Euler, zxy order, degrees) followed by a 180° turn about Y. Chosen out of 96
  convention combinations by the lowest brightness error in the overlap band (22.3 vs. 89.0 for the worst).
- The z angle (about 90°, the sensor mounting) is taken from lens 1 for both lenses. Using each lens's own z
  rotates lens 1 about its optical axis by the difference (0.77° on the X6) and leaves a vertical step at both
  seams. Measured with `re/lens_roll_est.py` on 9 clips (45 frames): residual roll −0.796° before, −0.042° after.
  Optical flow cannot absorb this: parallax runs across the seam, while a roll error runs along it.
- `trans` (3.27 cm lens spacing) is not used; near-field parallax is handled by the optical-flow seam instead.

## Projection model

The 13 distortion values fit the Mei omnidirectional model with radial-tangential and thin-prism terms:

```
p = (x, y) / (z + xi · |r|)          # unified sphere model
p = distort(p; k1..k5, p1 p2, p3 p4, s1..s4)
uv = p · f + c
valid where z / |r| ≥ cos(fov / 2) − 0.01
```

## Not verified

- Other models (X4, X5) and the `offset_v3` / `offset_v8` variants. `orbcut probe` reports
  `supported: false` for anything other than `offset_v6`.

## Reproduce

```bash
python re/search_convention.py   # rotation convention search
python re/render_test.py zxy     # equirect still
```
