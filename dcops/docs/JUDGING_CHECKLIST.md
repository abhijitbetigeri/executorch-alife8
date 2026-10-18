# DCSM-Simplified: judging checklist (Sat Oct 18, deadline 5:00 pm PDT)

One section per criterion: the evidence we have, where it lives, and what is still missing.

## Functionality & reliability (20)

Evidence:
- `build-fusion` runs on the board: camera -> NPU classifier -> info card, and mic -> NPU KWS -> checklist, in one HP image.
- Board top-3 matches PC top-3 on the same captured frame (background 66.4% vs 66.4%).
- Demo mask, held-out board frames, cutoff 0.5: 73% correct, 6% wrong, rest blank; 0% of background frames labelled.
- Clean held-out photos (5 classes, PC): 25/28.
- EMA (alpha 0.3) + confidence gate 0.5: blank instead of flicker.
- Known camera hang after J-Link reset is handled: power-cycle step; LED1 blinks if camera init fails.
- Flash guard refuses images past 0x80578000.

Gaps:
- Backup video: `TODO(record)`.
- A long soak run without hangs: not done.

## Performance & efficiency (20)

Evidence:

| Metric | Value |
|---|---|
| Vision NPU inference | 7.9–8.0 ms |
| Preprocessing (M55) | 1.75 ms |
| Full frame | 255–287 ms (3.5–3.9 FPS); LCD footer 274 ms / 3.6 FPS |
| KWS | MFCC 3.3–4.0 ms (M55) + DS-CNN 0.34 ms (NPU), 2 strides/s |
| Vision .pte / KWS .pte | 2301 KB / 93 KB |
| int8 vs float top-1 | 98.4% agreement |
| MRAM headroom | ~195 KB |

- NPU is ~6% of the 124 ms frame (was ~3% of 274 ms); bottleneck identified and the UART part removed; camera capture and LCD/LVGL remain. Same speed with or without a part on screen.
- Optimisation done: per-frame UART printing removed from the classify path.

Gaps:
- After the UART change: frame 124 ms, 8.0 FPS on the LCD footer (was 274 ms, 3.6 FPS): 2.2x faster. NPU stays 7.9 ms, so the NPU is now about 6% of the frame.
- Power: `TODO(measure)`. Not measured; needs a USB power meter or the power use case.

## Effective use of hardware (15)

Evidence:
- Both models on the Ethos-U85-256, 100% of graph ops; only quantize/dequantize on the M55.
- M55 HP does crop/resize, MFCC, softmax/EMA, UI.
- Camera MIPI/CPI + DMA, PDM mic, LCD, MRAM for models and info cards.
- Memory map and boot-table guard (`ARCHITECTURE.md`).
- Placement table with measured costs (`ARCHITECTURE.md`, `DEVPOST.md`).

Gaps:
- HE core (M55 + Ethos-U55) unused. KWS on HE + U55 is documented as the next step, not built.

## Multimodal & sensor integration (15)

Evidence:
- Camera and microphone run concurrently in one image, two ExecuTorch programs (per-instance `EtModel`).
- Fusion: `go` only works when vision is confident; `no` on item 1 overrides vision.
- KWS trained on the firmware's exact MFCC with board AGC emulated: 94.66% test, 94.70% with AGC.
- Keyword-to-action mapping documented (`DEVPOST.md`, `ARCHITECTURE.md`).

Gaps:
- None open beyond the demo itself working in a noisy room (fallback in `DEMO_SCRIPT.md`).

## Innovation & use-case value (10)

Evidence:
- Air-gapped data-center use case: no phones, no cloud.
- Info cards plus a hands-free inspection checklist.
- No-network board-camera data collection (clock-synced slideshow).
- Future work scoped with numbers: component-level parts need a detector or tiled inference (resistor ~2 px at 160 px input).

Gaps:
- None.

## Technical ambition & integration (10)

Evidence:
- YOLOv8n-seg -> ConvNeXt teachers (97.2%) -> MobileNetV2 student (0.912 val / 0.913 board-sim).
- Ablations: no board-camera aug 0.758, no KD 0.804 on board-sim. Resolution sweep 128/160/192 px.
- Board-camera fine-tune: held-out board 35.5% -> 73.0%.
- AST -> DS-CNN distillation.
- MLEK runtime change for two models; Vela miscompile found and worked around (`docs/blazeface-vela-bug.md`).

Gaps:
- None.

## Engineering quality & demo (10)

Evidence:
- Public repo: https://github.com/abhijitbetigeri/executorch-alife8 (fork of npow/alif-mlek-yolo; prior work disclosed).
- Scripts: `dcops/jobs/`, `dcops/colab/`, `dcops/export/`, `dcops/board_capture/`, `scripts/alif/build_dcops_demo.sh`, guarded `scripts/alif/flash_e8_hp.sh`.
- Docs: `DEVPOST.md`, `ARCHITECTURE.md`, `RESULTS.md`, `README_SECTION.md`, `DEMO_SCRIPT.md`.
- 2-minute run sheet with fallbacks.

Gaps:
- Judged tag: `TODO(tag)`, created by a human after review.
- Team names: `TODO(names)`.
- Demo rehearsed twice: not yet.

## Submission form must-haves

- [x] Public repo URL.
- [ ] Commit hash / tag of the judged build: `TODO(tag)`.
- [x] README with build / export / flash / run (`README_SECTION.md`).
- [x] Architecture and workload placement (`ARCHITECTURE.md`, `DEVPOST.md`).
- [ ] Perf: latency and memory done; FPS after UART change and power are `TODO(measure)`.
- [x] Disclosure of reuse (DC-Ops phone project, npow/alif-mlek-yolo camera pipeline).
- [ ] Demo video: `TODO(record)`.
- [ ] Team names: `TODO(names)`.

## Remaining timeline (Oct 18)

| Time | Milestone |
|---|---|
| 13:30 | Read FPS from the LCD footer on the current build; fill it in |
| 14:00 | Record the backup video |
| 15:00 | Docs/README merged, tag created, Devpost draft complete |
| 16:00 | Rehearse the demo twice; submit by **16:30** (30 min buffer) |
