# Face detection with ExecuTorch on the Alif E8 (BlazeFace)

`alif_object_detection` builds with either ML framework:

| `-DML_FRAMEWORK=` | Model | Input | Runtime |
|---|---|---|---|
| `TensorFlowLiteMicro` (default) | YOLO Fastest face, `yolo-fastest_192_face_v4_vela_Z256.tflite` | 192×192 grayscale | TFLM + Ethos-U |
| `ExecuTorch` | MediaPipe BlazeFace front, `resources/object_detection/blazeface/blazeface_front_128_ethos-u85-256.pte` | 128×128 RGB | ExecuTorch 1.1.0 + Ethos-U85 delegate |

The camera, preview and box drawing are shared (see [alif-e8-camera.md](alif-e8-camera.md)); only the model, its pre-processing and its post-processing change.

## Architecture and workload placement

```
MT9M114 (MIPI, 1280x720 RGB565)
  │  CPI + DMA into SRAM
  ▼
Cortex-M55 HP: rotate 180°, full-screen RGB888 preview for the LCD,
               centre-crop + resize to 128x128 RGB888           (camera HAL)
  │
  ▼
Cortex-M55 HP: RGB888 bytes -> float [0,1] NHWC                 BlazeFacePreProcess
  │
  ▼
Cortex-M55 HP: float -> int8 (cortex_m::quantize_per_tensor)    ExecuTorch, CPU
Ethos-U85-256: all 97 BlazeFace operators (one Vela command stream)  ExecuTorch EthosUBackend
Cortex-M55 HP: int8 -> float (cortex_m::dequantize_per_tensor)  ExecuTorch, CPU
  │  scores 1x896, boxes 1x896x16
  ▼
Cortex-M55 HP: anchor decode, sigmoid threshold, weighted NMS    BlazeFacePostProcess
  │
  ▼
CDC200 + MIPI DSI: LVGL draws the preview and green face boxes on the ILI9806E LCD
```

| Workload | Where | Code |
|---|---|---|
| Camera capture | CPI + DMA | `source/hal/source/components/camera/` |
| Debayer/rotate/resize, preview | M55 HP | `image_processing.c`, `hal_camera_alif.c` |
| Input conversion | M55 HP | `source/lib/mlek/use_case/object_detection/BlazeFaceProcessing.cc` |
| BlazeFace network | Ethos-U85 (100% of operators) | `.pte` via `EtModel` |
| int8 boundary | M55 HP (Cortex-M kernels) | ExecuTorch `cortex_m` ops |
| Box decode + NMS | M55 HP | `BlazeFaceProcessing.cc` (anchors generated, not stored) |
| Display | CDC200 + DSI, LVGL | `alif_ui`, `lvgl_port` |

## Build

Requirements: Arm GNU toolchain (14.3.1 tested; the build log shows only upstream warnings), CMake ≥ 3.21, the submodules (`git submodule update --init --recursive`), and a Python environment with ExecuTorch 1.1.0 at `resources_downloaded/env` (the ExecuTorch runtime is compiled from `dependencies/executorch`, and its codegen runs in that environment; `flatc` must be in its `bin/`). See [the BlazeFace README](../resources/object_detection/blazeface/README.md) for the exact environment.

The configure step also checks `resources_downloaded/resources_downloaded_metadata.json`, which `set_up_default_resources.py` writes. That script also downloads every TFLM demo model. For an ExecuTorch-only build you can instead write the file by hand with the MD5 of `set_up_default_resources.py`:

```sh
mkdir -p resources_downloaded && ln -sfn <path-to-et-1.1.0-venv> resources_downloaded/env
python3 -c "import hashlib,json; json.dump({'set_up_script_md5sum': hashlib.md5(open('set_up_default_resources.py','rb').read()).hexdigest()}, open('resources_downloaded/resources_downloaded_metadata.json','w'))"
```

```sh
cmake -S . -B build-blazeface \
  -DCMAKE_TOOLCHAIN_FILE=scripts/cmake/toolchains/bare-metal-gcc.cmake \
  -DTARGET_PLATFORM=alif -DTARGET_SUBSYSTEM=RTSS-HP \
  -DTARGET_BOARD=DevKit-e8 -DUSE_CASE_BUILD=alif_object_detection \
  -DML_FRAMEWORK=ExecuTorch -DETHOS_U_NPU_ID=U85 -DETHOS_U_NPU_CONFIG_ID=Z256 \
  -DALIF_CAMERA_ENABLED=ON -DALIF_CAMERA_MODULE=MT9M114 -DALIF_ISP_ENABLED=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-blazeface --target mlek_alif_object_detection -j8
```

Options: `-Dalif_object_detection_SCORE_THRESHOLD=0.75` (face score), `-Dalif_object_detection_MODEL_PATH=<.pte>`, `-Dalif_object_detection_ACTIVATION_BUF_SZ=0x00100000`.

Flash exactly as in [alif-e8-camera.md](alif-e8-camera.md), with `build-blazeface/bin/mlek_alif_object_detection.axf`.

## Performance

The console prints, every 30 frames:

```
BlazeFace avg over 30 frames: pre X ms, NPU inference X ms, post X ms, frame-to-frame X ms (X FPS)
```

| Metric | BlazeFace / ExecuTorch | YOLO Fastest / TFLM |
|---|---|---|
| Operators on the NPU | 97 / 97 | (Vela report) |
| Model size | 189 KB | (tflite) |
| NPU SRAM (Vela) | 387 KB | |
| MACs per frame | 31.7 M | |
| NPU inference (incl. int8 boundary) on DevKit-E8 | **4.61 ms** | not measured |
| Pre-processing (RGB888 to float) | 1.08 ms | not measured |
| Post-processing (decode + NMS) | 0.05 ms | not measured |
| Frame-to-frame (camera + preview + model + LVGL) | 166 ms, **6.0 FPS** | not measured |

## Verification so far

- Export: all operators delegated to the Ethos-U85; int8 detections match float on reference photos.
- Post-processing: `BlazeFaceProcessing.cc` compiled on the host and compared with the reference PyTorch implementation on the same model outputs: same faces, boxes within 1 px.
- Firmware: builds for DevKit-E8 (GCC 14.3.1, Release) on macOS arm64 to `build-blazeface/bin/mlek_alif_object_detection.axf`, with the ExecuTorch runtime, the Ethos-U backend, BlazeFace pre/post-processing and the 195 KB program linked in.
- On the board (2026-10-17, DevKit-E8, M55 HP at 400 MHz): flashed with J-Link and running. Averages over 26 frames, read from the firmware's counters over J-Link: NPU inference 4.61 ms, pre-processing 1.08 ms, post-processing 0.05 ms, 166 ms frame-to-frame (6.0 FPS). The face model is 3.4% of the frame time; the rest is camera capture, the full-screen preview resize and LVGL drawing on the M55. Detection quality on live faces: not yet checked.
- Console: this build prints on UART2 (`CONSOLE_UART` default), which the DevKit's PRG USB bridge does not carry. Configure with `-DCONSOLE_UART=4` and set SW4 to UART4 to see the timing lines on the USB serial port. Until then, read the counters with J-Link (`timingFrames`, `infCycles`, `preCycles`, `postCycles`, `frameCycles` in the map file).

### Firmware memory (Release build)

| Region | Used | Contents |
|---|---|---|
| MRAM (code + constants) | ~1.1 MB of 4.5 MB | startup/code 698 KB, read-only 230 KB (incl. the 195 KB `.pte`), ITCM code image 163 KB |
| Bulk SRAM `.bss.sram` | 6.4 MB | camera frame buffers, full-screen preview, LVGL, 1 MB ExecuTorch arena |
| `.bss.unstriped` | 750 KB | display framebuffer |
| DTCM | ~300 KB | `.bss`, heap 92 KB, stack 32 KB |

## Update: scrambled outputs fixed (not yet re-run on the board)
The board's scrambled BlazeFace outputs were a Vela miscompile of cancelling transposes before a reshape; see [blazeface-vela-bug.md](blazeface-vela-bug.md). The committed `.pte` is now the patched export (head tail on the CPU), bit-exact on a Corstone-320 FVP. Re-flash and re-check on the board.
