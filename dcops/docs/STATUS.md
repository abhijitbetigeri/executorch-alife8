# DC-Ops on E8: status (13:15, 2026-10-18)

The demo works on the board. What remains is one FPS reading, the video, the tag and the submission.

## Done
- **Firmware on the board** (`build-fusion`, one HP image): camera classifier + info cards + voice checklist, both models on the Ethos-U85. Image ends at 0x805473C4, ~195 KB below the 0x80578000 guard.
- **Vision model** `v3_board_ft` (deployed): board-camera fine-tune of `v2_r160_kd`. Held-out board frames, all classes: 35.5 % -> 73.0 %. .pte 2301 KB, int8/float agreement 98.4 %.
- **Demo mask** `CLS_ALLOWED_MASK=0x13250` (power shelf, network port, fan, power connector, drive bay + background), cutoff 0.5: 73 % correct, 6 % wrong, rest blank; background never labelled.
- **Voice** `dscnn_m_agc`: 94.66 % test (94.70 % with board AGC), 93 KB. `go` / `yes` / `no` / `up` / `down` / `stop` drive the checklist.
- **On-board correctness**: board top-3 = PC top-3 on the same frame (background 66.4 % vs 66.4 %).
- **Measured**: vision NPU 7.9–8.0 ms, preprocessing 1.75 ms, frame 255–287 ms (3.5–3.9 FPS), KWS MFCC 3.3–4.0 ms + DS-CNN 0.34 ms. (With the earlier v2 model, same architecture.)
- **Optimisation**: per-frame UART result/profiler printing removed from the classify path; flashed.
- **Vela bug**: miscompile found and worked around (`docs/blazeface-vela-bug.md`).
- **Docs** (`dcops/docs/`): Devpost text, architecture, results, demo run sheet, judging checklist, README section.

## To do
1. Done: judged build reads frame 124 ms, 8.0 FPS (was 274 ms, 3.6 FPS).
2. Record the backup demo video: `TODO(record)`.
3. Commit `scripts/alif/build_dcops_demo.sh` (being added) and the docs; tag the judged commit: `TODO(tag)`.
4. Team names on Devpost: `TODO(names)`.
5. Rehearse the 2-minute demo twice; submit before **5:00 pm PDT** (target 16:30).

## Not doing
- Power measurement: not measured (`TODO(measure)`); say so if asked.
- KWS on the HE core + Ethos-U55: documented next step, not built.

## Reminders
- After every flash, power-cycle (unplug PRG USB about 10 s). LED1 blinking = camera init failed.
- Flash only with `scripts/alif/flash_e8_hp.sh`; it refuses images past 0x80578000.
