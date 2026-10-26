# Stitch quality log — closing the gap to Insta360 Studio

How `orbcut`'s own stitcher (factory calibration + optical-flow seam) was measured against
Insta360 Studio 6.0.6 on the same clip, and what actually moved the numbers. Kept as a record so the
reasoning survives — including the hypotheses that turned out wrong.

Reference clip: Insta360 X6, 50 s, night-time bar interior, people 1–2 m from the camera, hand-held.
Studio export: 360 equirect, "optical flow" stitching, 10-bit HEVC. Ours: `python -m orbcut.render panorama`
at 7680×3840. Everything below is measured on the same 10 s window of that clip (frames 10–20 s).

## 1. Measurement

Two equirect videos cannot be compared pixel-for-pixel: Studio applies its own horizon lock, so its
orientation drifts frame by frame relative to our raw camera frame.

`re/studio_compare.py` therefore:

1. Grabs frames at the same timestamps, scaled to 3840×1920.
2. Estimates the rotation between them from ORB features converted to unit vectors (RANSAC Kabsch),
   and rotates ours into Studio's frame.
3. Splits the sphere into a *seam band* (directions within ±6° of being perpendicular to the lens
   axis, |lat| < 60°) and the *body* (the rest, |lat| < 60°).
4. Reports, per region, Laplacian variance (sharpness), residual noise in flat areas, SSIM and Lab means.

Caveat (confirmed later): Laplacian variance rewards noise and hard cut lines as much as real detail.
It is fine for *body* comparisons and for the final encoded output, but it is not enough for tuning the
seam. For that `re/seam_ab.py` renders frames straight from the source (no encode) and measures, in our own
coordinates on the blend band (0.2 < α < 0.8):

| metric | definition | reads as |
|---|---|---|
| `ret` | Σ‖∇G₁(blend)‖ / max(Σ‖∇G₁(lens0)‖, Σ‖∇G₁(lens1)‖) | high-frequency retention; 1.0 = blending lost nothing |
| `fake` | mean max(0, ‖∇blend‖ − max(‖∇lens0‖, ‖∇lens1‖)) / mean max(…) | edges that exist in neither lens: ghosts or cut lines |
| `flick` | mean |frame t − frame t+1| in band ÷ same in body | temporal instability relative to the body |
| `wob_t` | mean ‖F_t − F_{t−1}‖ (px) | frame-to-frame change of the flow field = visible "wobble" |
| `wob_s` | mean ‖F − G₈(F)‖ (px) | spatial high-frequency of the flow = ripples on straight edges |

An early version of `ret` used the pixel-wise max of the two lens gradients as denominator. That
picks the noisier lens at every pixel, so even a single-lens image scores ~0.6. Use the sum-then-max form.

## 2. Timeline of changes and what each did

Numbers are Studio = 1.0 unless stated. "seam"/"body" = Laplacian ratio in the two regions of the
*encoded* output; `ret`/`fake`/`wob` are frame-level.

### Baseline (before any work)

| | seam | body | noise | ΔL |
|---|---|---|---|---|
| baseline | 0.53 | 0.72 | 1.31 | −4.2 |

Softer everywhere, noisier, and darker.

### Resampling: bilinear → bicubic, double remap → single remap

The flow-corrected seam was sampled twice (source → warped lens image → flow-warped result). Composing the
flow into the coordinate maps and sampling the source once, with bicubic interpolation, gave

| | seam | body | noise |
|---|---|---|---|
| cubic + single sampling | 0.63 | 0.87 | 1.47 |

Body sharpness recovered most of the gap; noise got *worse* because less interpolation means less
implicit smoothing. Pre-encode frames at this point were already 0.98 (seam) / 1.24 (body) — i.e. the
encoder was throwing most of it away.

### Encoding: what the encoder was doing

Five finish-filter variants were encoded and compared. The combination that won:

- `hqdn3d=6:4.5:9:6.75` (mild temporal denoise) then `unsharp=5:5:0.6:5:5:0` (mild sharpen) before the encoder.
  Rationale: with the noise left in, the HEVC encoder spends its bits on noise and drops detail.
- Colour: the decode/encode path used ffmpeg's default BT.601 matrix on untagged frames. Forcing
  `in_color_matrix=bt709` on decode and `out_color_matrix=bt709` + tags on encode removed the −4 L* offset.
- 10-bit (`p010le`, `main10`) and constant quality (`-q:v 70`) instead of a fixed bitrate.

| variant | seam | body | noise | ΔL |
|---|---|---|---|---|
| A2 no filter, BT.709 | 0.65 | 0.77 | 0.82 | −1.5 |
| F  + hqdn3d/unsharp | 0.71 | 0.90 | 0.26 | −1.8 |
| G  + 10-bit | 0.67 | 0.94 | 0.19 | −1.3 |
| H  + `-q:v 70` | 0.73 | 0.94 | 0.09 | −1.3 |

H became the default. The 10 s segment is 97 Mbps vs Studio's 121 Mbps at this quality.

### Seam: two external reviewers, one A/B, and the hypotheses that failed

Two LLM reviewers (asked blind, with the pipeline description only) both ranked "the feather band is too wide"
as the primary cause and suggested, in various orders: narrow feather, remove the row tiling around the seam,
raise the elevation limit for flow, multi-band blending, confidence-gated winner-take-all, forward/backward
flow checks.

Frame-level A/B (`re/seam_ab.py`, 5 timestamps × 2 consecutive frames):

| variant | ret | fake | flick | Laplacian vs Studio |
|---|---|---|---|---|
| base (feather, flow, 256-row tiles) | 0.88 | 0.157 | 1.09 | 1.14 |
| flow EMA off | 0.88 | 0.157 | 1.09 | 1.14 |
| feather 0.12 → 0.04 | 0.86 | 0.132 | 1.05 | 0.99 |
| flow elevation limit 55° → 70° | 0.89 | 0.161 | 1.09 | 1.14 |
| no flow at all | 0.81 | 0.002 | 0.97 | 0.95 |
| vertical strip instead of tiles | 0.90 | 0.163 | 1.05 | 1.17 |
| **two-band + confidence (tiles)** | 0.96 | 0.182 | 1.13 | 1.66 |
| **two-band + confidence (strip)** | 0.98 | 0.186 | 1.10 | 1.77 |

What this says:

- Feather width does not matter. Narrowing it slightly *hurts*. The "wide feather is the culprit" hypothesis
  was wrong for this pipeline.
- Flow is clearly worth having (`ret` 0.81 → 0.88) but it is also where the ghosts come from (`fake` 0.002 → 0.157).
- The only change that moved `ret` substantially is **two-band blending**: blend the low frequencies with the
  wide α (hides exposure/vignetting steps) and switch the high frequencies over a narrow transition
  (`α_hi = clip((α − 0.5)·k + 0.5)`, k = 10). Residual flow error of 1–2 px cancels high-frequency phase when
  two images are averaged; if only one image contributes the high frequencies, there is nothing to cancel.
- Confidence gating on top of that: forward/backward flow consistency (> 1.5 px) and post-warp residual
  (> 24/255) mark pixels where the flow is not trustworthy (occlusion, no correspondence); there the high
  frequencies come from one lens only. Small but positive (`ret` 0.952 → 0.957 in isolation).

Bug found on the way: the seam processing regions were built per 256-row tile, separately for "flow" and
"feather-only" (polar) zones, and both wrote the whole α-band mask of the tile. Where a tile straddled the
55° elevation boundary — and for the full-height strip, everywhere — the feather region overwrote the flow
result. Symptom: the "strip" variant produced numbers identical to "no flow". Fixed by masking each region
with its own zone.

### Encoded result after two-band

| | seam | body | noise | ΔL |
|---|---|---|---|---|
| two-band, strip | **1.20** | 0.95 | 0.08 | −1.3 |

### Wobble: the one the numbers did not show

Viewed side by side (`re/seam_video.py`, Studio re-oriented per frame to our coordinates), the seam band in our
output visibly rippled while Studio's did not — even though every sharpness metric said we were ahead.
Two new metrics (`wob_t`, `wob_s`) made it measurable:

| variant | wob_t (px/frame) | wob_s (px) | ret | flick |
|---|---|---|---|---|
| two-band as above | 6.09 | 0.75 | 0.983 | 1.095 |
| + spatial smoothing of flow σ=32 px | 4.35 | 0.12 | 0.979 | 1.088 |
| + clamp vertical flow to ±3 px | 4.25 | 0.22 | 0.979 | 1.076 |
| + fixed temporal EMA 0.8 | **0.79** | **0.09** | 0.977 | 1.074 |
| (fixed EMA 0.9) | 0.44 | 0.18 | 0.979 | 1.076 |
| (magnitude clamp 40 px) | 1.84 | 0.19 | 0.961 | 1.075 |

Root cause of the 6 px/frame figure: a "motion-adaptive" EMA had been added that drops the temporal
smoothing wherever the flow changes by more than σ = 2 px between frames. DIS flow noise on this footage is
~6 px, so the adaptive rule disabled smoothing almost everywhere. A fixed EMA of 0.8 plus spatial smoothing
removed the ripple with no measurable loss in sharpness; the user confirmed visually. The magnitude clamp
was rejected: it cuts real near-field parallax and costs retention.

Final (encoded, 5 s window): seam 1.23, body 0.97, noise 0.04, ΔL −1.2.

## 3. Current seam pipeline (for orientation)

1. Static per-lens sample maps from the factory calibration (`re/README.md`).
2. Blend weight α from cos(angle to optical axis), linear feather of width 0.12 (cosine units).
3. One vertical strip per seam, covering only the rows where |elevation| < 55° (poles are feather-only).
4. Both lenses warped to the output grid at 0.5× → DIS optical flow (medium preset) lens0 → lens1,
   brightness-normalised; Gaussian-smoothed σ = 32 px (output scale); vertical component clamped to ±3 px;
   temporal EMA 0.8.
5. Backward flow for a forward/backward consistency map; post-warp residual; combined confidence,
   smoothed, temporally averaged.
6. Flow composed into the coordinate maps (lens0 by −α·F, lens1 by +(1−α)·F); single bicubic sample.
7. Two-band blend: `out = I0 + (I1 − I0)·α_hi + (L1 − L0)·(α − α_hi)` with L = Gaussian σ 6 (computed at
   ¼ resolution), α_hi = narrow sigmoid of α, further hardened to one lens where confidence is low.
8. Encode: BT.709 in/out, `hqdn3d` + `unsharp`, 10-bit HEVC, constant quality.

Every knob is an environment variable (`ORBCUT_FEATHER`, `ORBCUT_FLOW_EMA`, `ORBCUT_FLOW_SMOOTH`,
`ORBCUT_FLOW_VCLAMP`, `ORBCUT_SEAM_MODE`, `ORBCUT_BAND2_K`, …) so A/B runs do not need code changes.

## 3a. Lens 1 roll (fixed 2026-10-26)

A straight vertical step kept showing at both seams on moving footage. Optical-flow measurements in a strip
fixed to the camera showed the two lenses disagreeing *along* the seam by a constant −0.80° on every clip
(9 clips, 45 frames, per-clip −0.78 to −0.82°). Parallax only moves things *across* the seam, so this was a
calibration-reading error, not depth. Using lens 1's z angle (sensor mounting, ~90°) for both lenses removes it:
the factory values differ by 0.77°, and the residual drops to −0.04°. Optical flow could not absorb it because
the vertical correction is deliberately clamped to ±3 px to keep the seam from wobbling.

| | before | after |
|---|---|---|
| lens 1 roll residual (9 clips) | −0.796° | −0.042° |
| seam structural similarity vs. Studio | 0.722 | 0.765 |
| seam / body sharpness vs. Studio | 1.23 / 0.97 | 1.23 / 0.98 |
| moving-view lens misalignment (0–255) | 19.5 | 17.4 |

## 3b. Dynamic renders (tracking views, panorama swap)

Views whose orientation changes every frame (`DynamicProjector`) used to blend the overlap with the
weight feather only, because a per-view flow history is meaningless once the view rotates. A person sitting
on the seam showed as a transparent double image. The fix estimates the flow in a strip fixed to the camera:
a ring perpendicular to lens 0's optical axis, θ along the seam and φ across it (the parallax direction).
The strip does not move when the view rotates, so the same temporal averaging as the static renderer holds.
Each frame only the strip columns visible in the view are computed; the displacement is composed into the
sampling directions (lens 0 by −α·F, lens 1 by +(1−α)·F) and the two-band blend follows.

Measured on a 32 s tracked clip (70°, 1920×1080, 958 frames, 44 samples):

| | feather | strip flow + band2 |
|---|---|---|
| lens-to-lens misalignment in the blend band (0–255) | 23.2 | 19.5 |
| high-frequency retention | 0.76 | 0.93 |
| fake edges | 0.00 | 0.12 |
| strip flow change per frame (strip px) | – | 0.20 |
| face sharpness vs. 4K panorama preview | ×19.9 | ×37.9 |
| processing per frame | 0.03 s | 0.15 s |

## 4. Known limits

- Objects 1–2 m from the camera have real parallax that a rotation-only model cannot resolve. Where the
  flow is unreliable the high frequencies simply come from one lens, which trades a ghost for a small
  geometric step. Fixing that properly needs depth, which is out of scope.
- Studio's stabilisation and horizon lock are not part of this comparison (`orbcut.track` handles that
  separately from the IMU).
- VideoToolbox allows a limited number of concurrent HEVC encoder sessions (measured on an M2 Ultra:
  4 for 10-bit constant-quality, 5 for 10-bit bitrate, 6+ for 8-bit). A second process using the encoders
  makes ours fail with `-12912`; `orbcut.render views` now encodes in waves of 4.

## 5. Reproduce

```bash
# encoded comparison on a 5 s window (needs the Studio export next to the source)
python tests/check_studio_compare.py            # or --remote
# frame-level seam A/B across variants
python re/seam_ab.py <clip.insv> <studio.mp4> <out_dir> --times 3,10,17,24,31
# side-by-side video for eyes
python re/seam_video.py <studio.mp4> <ours.mp4> <out.mp4> --start 10 --dur 10 --offset 10
```
