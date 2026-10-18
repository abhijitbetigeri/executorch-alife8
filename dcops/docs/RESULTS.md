# DC-Ops on E8: training results

All numbers are offline (PC) evaluations, produced on 2026-10-18 by `dcops/jobs/` on a Colab L4. On-board numbers: run `dcops/board/measure.py`.

## Teachers

| | crops | teacher | accuracy |
|---|---|---|---|
| wave 1 | 3,494 (YOLO pseudo-labels on 319 images) | ConvNeXt-Small | 69.4 % (val, auto labels) |
| wave 2 | 19,203 (2,056 human-labelled images + wave-1 crops for missing classes) | 2x ConvNeXt-Base, out-of-fold | **97.2 %** human-labelled val; 94.0 % out-of-fold train |

## Vision students (MobileNetV2, 160 px unless named, all ops on the Ethos-U85)

| run | metric | val | board-camera sim | int8 = float | .pte KB |
|---|---|---|---|---|---|
| r128_kd | val (auto labels) | 0.581 | 0.560 | 1.0 | 2302 |
| r160_kd | val (auto labels) | 0.653 | 0.593 | 0.98 | 2303 |
| r160_kd_e120 | val (auto labels) | 0.644 | 0.636 | 0.95 | 2306 |
| r160_noaug | val (auto labels) | 0.584 | 0.498 | 0.95 | 2301 |
| r160_nokd | val (auto labels) | 0.667 | 0.536 | 0.95 | 2304 |
| r192_kd | val (auto labels) | 0.632 | 0.531 | 0.97 | 2301 |
| v2_r160_kd | human-labelled val | 0.912 | 0.913 | 1.0 | 2302 |
| v2_r160_kd_mix | human-labelled val | 0.914 | 0.814 | 0.98 | 2301 |
| v2_r160_kd_mix_e80 | human-labelled val | 0.927 | 0.839 | 1.0 | 2304 |
| v2_r160_noaug | human-labelled val | 0.923 | 0.758 | 1.0 | 2295 |
| v2_r160_nokd | human-labelled val | 0.912 | 0.804 | 1.0 | 2304 |

Wave 1 (rows without `v2_`) is scored on noisy auto labels; wave 2 on the human-labelled split, so the two are not directly comparable.
Ablations (wave 2): board-camera augmentation 0.913 vs 0.758 on board-like images (`v2_r160_kd` vs `v2_r160_noaug`); distillation 0.913 vs 0.804 (`v2_r160_kd` vs `v2_r160_nokd`). Board-sim scores are a single random draw per evaluation, so differences of a few points are noise.
**Deployed:** `v2_r160_kd` -> `resources/object_detection/dcops/`.

## Keyword spotters (DS-CNN, MLEK MFCC 49x10, 12 classes)

| run | test | test with board auto-gain | params | int8 = float | .pte KB |
|---|---|---|---|---|---|
| dscnn_m_agc | 0.9466 | 0.9470 | 79116 | 0.88 | 93 |
| dscnn_m_kd | 0.9444 | - | 79116 | 0.89 | 93 |
| dscnn_s_agc | 0.9394 | 0.9409 | 23180 | 0.89 | 34 |
| dscnn_s_kd | 0.9366 | - | 23180 | 0.88 | 33 |

**Deployed:** `dscnn_m_agc` -> `resources/kws/dcops/` (trained with the board's auto-gain emulated).

## Firmware (build-fusion: camera classifier + voice, one HP image)

MRAM end 0x805475C4 (194 KB below the 0x80578000 boot-table guard); vision .pte 2302 KB, KWS .pte 93 KB. Not yet run on the board.
