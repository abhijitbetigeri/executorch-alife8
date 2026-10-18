# DC-Ops on E8: status (morning of 2026-10-18)

Everything that can run without the board is done; the repo is pushed. What remains needs the board.

## Done
- **Models** (in `resources/`, all operators on the Ethos-U85):
  - Vision: MobileNetV2 `v2_r160_kd`, 91.2 % on the human-labelled validation split, 0.913 on simulated board-camera frames; 2.3 MB.
  - Voice: DS-CNN `dscnn_m_agc`, 94.7 % test accuracy, unchanged with the board's auto-gain simulated; 93 KB.
- **Firmware** (`build-fusion`): camera classifier + voice in one Cortex-M55 HP image, 194 KB below the boot-table guard. Voice drives the scanner ("go" holds the card and opens a checklist, "stop" resumes, "up"/"down"/"yes"/"no" work the checklist); info cards and a performance footer on the LCD.
- **Board tools** (`dcops/board/`): guarded flash with a boot-area check, J-Link measurement, board-vs-PC snapshots; tested offline.
- **BlazeFace bug**: root cause is a Vela compiler miscompile; workaround committed and verified on a Corstone-320 FVP (`docs/blazeface-vela-bug.md`).
- **Docs** (`dcops/docs/`): Devpost text, architecture, demo script, judging checklist, results, recovery steps.

## To do (needs the board)
1. Flash `build-fusion` and power-cycle. Nothing from the overnight run has run on the board yet.
2. Verify on hardware: the camera classifier on real components, the mic and keywords, readability of the info card and checklist.
3. Measure NPU latency, FPS with voice running, and memory; fill the placeholders in `DEVPOST.md`.
4. Fill in the measured numbers, tag the release commit, record a backup demo video, and submit on Devpost before **5:00 pm PDT, Oct 18**.

## Open question
Board-camera accuracy will be lower than the PC numbers. If it is weak on the actual demo objects, capture a few minutes of board-camera frames of those objects and run a quick fine-tune (about 30 minutes on Colab).

## Start here
```sh
cd ~/projects/arm-executorch-hack/executorch-alife8 && source ../env.sh
dcops/board/flash.sh build-fusion/bin/mlek_alif_object_detection.axf   # then unplug/replug the board
python3 dcops/board/measure.py build-fusion/bin/mlek_alif_object_detection.axf --seconds 20
dcops/.venv-train/bin/python dcops/board/snapshot.py build-fusion/bin/mlek_alif_object_detection.axf --lcd
```
If keywords never register, rebuild with `-DTARGET_MICS=I2S`.
