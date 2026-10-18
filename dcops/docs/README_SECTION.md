## DC-Ops on E8: build, export, flash, run

A handheld, air-gapped data-center component identifier. The camera feeds a MobileNetV2 on the
Ethos-U85, and the LCD shows the component and its info card. Voice keywords drive the UI. It ports our
earlier phone project [DC-Ops](https://github.com/abhijitbetigeri/DC-Ops), whose dataset, YOLOv8n-seg
detector and info-card text we reuse. The distillation pipeline, ExecuTorch/Ethos-U85 export and E8
firmware are new. Details: [`dcops/docs/ARCHITECTURE.md`](dcops/docs/ARCHITECTURE.md).

### Prerequisites

- Arm GNU toolchain 14.3.1, CMake ≥ 3.21, `git submodule update --init --recursive`
- ExecuTorch 1.1.0 Python environment at `resources_downloaded/env`. See
  [docs/blazeface-executorch.md](docs/blazeface-executorch.md#build) for the venv and
  `resources_downloaded_metadata.json` setup.
- SEGGER J-Link (on-board J-Link on the PRG USB port)

### 1. Train (optional, GPU)

```sh
# teacher crops + ConvNeXt teacher, then students and keyword spotters (HF Jobs; or use the Colab notebook)
hf jobs uv run --flavor a10g-small --timeout 3h --secrets HF_TOKEN dcops/jobs/teacher.py
hf jobs uv run --flavor a10g-small --timeout 4h --secrets HF_TOKEN dcops/jobs/student.py --name r160_kd --res 160 --kd-alpha 0.7
hf jobs uv run --flavor a10g-small --timeout 5h --secrets HF_TOKEN dcops/jobs/kws.py --name dscnn_s_kd
```

All runs at once: `dcops/launch_overnight.sh` or `dcops/colab/dcops_overnight.ipynb`.

### 2. Export to ExecuTorch for the Ethos-U85-256

```sh
# run folder contains student.pt, metrics.json, calib.npz, labels.txt
resources_downloaded/env/bin/python dcops/export/export_student.py <run-dir>   # -> <run-dir>/student_ethos-u85-256.pte + export.json
resources_downloaded/env/bin/python dcops/export/export_kws.py     <kws-run-dir>
```

Check `export.json`: `cpu_ops` should list only the quantize/dequantize boundary, and
`float_int8_agreement` should be close to 1.

### 3. Build the firmware (M55 HP)

```sh
cmake -S . -B build-dcops \
  -DCMAKE_TOOLCHAIN_FILE=scripts/cmake/toolchains/bare-metal-gcc.cmake \
  -DTARGET_PLATFORM=alif -DTARGET_SUBSYSTEM=RTSS-HP -DTARGET_BOARD=DevKit-e8 \
  -DUSE_CASE_BUILD=alif_object_detection \
  -DML_FRAMEWORK=ExecuTorch -DETHOS_U_NPU_ID=U85 -DETHOS_U_NPU_CONFIG_ID=Z256 \
  -DALIF_CAMERA_ENABLED=ON -DALIF_CAMERA_MODULE=MT9M114 -DALIF_ISP_ENABLED=OFF \
  -Dalif_object_detection_ET_TASK=CLASSIFY \
  -Dalif_object_detection_MODEL_PATH=<run-dir>/student_ethos-u85-256.pte \
  -Dalif_object_detection_CLASS_LABELS=<run-dir>/labels.txt \
  -Dalif_object_detection_ACTIVATION_BUF_SZ=0x00180000 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-dcops --target mlek_alif_object_detection -j8
```

`TODO(confirm)`: extra options for the info card and KWS once they are merged.
Add `-DCONSOLE_UART=4` (and set SW4 to UART4) to see the timing log on the USB serial port.

### 4. Flash, using the guard script

```sh
scripts/alif/flash_e8_hp.sh build-dcops/bin/mlek_alif_object_detection.axf
```

The HP app partition is **0x80200000–0x80580000**, and Alif's ATOC boot table sits at its top
(0x8057BC00). The linker accepts larger images, but flashing one stops the HP core from booting.
The script refuses any image ending past **0x80578000**. Do not flash with a raw `loadfile` script.
If boot is lost anyway, restore an MRAM backup of the 0x80200000 partition by writing it through the HE core (J-Link HE device name: `TODO(confirm)`; the procedure used on Oct 17 is not yet written down), since the HP core no longer connects.

### 5. Power-cycle, then run

**After every flash, unplug the PRG USB cable for a few seconds and plug it back in.** The MT9M114
camera hangs after a J-Link reset, so the preview stays frozen or black until the board is power-cycled.

The LCD shows the camera preview with the top-2 labels. Point it at server hardware, or at a photo
on a laptop screen, from 20–30 cm away, with the component in the centre of the frame.

### Performance

See the tables in [`dcops/docs/DEVPOST.md`](dcops/docs/DEVPOST.md#performance). Timing counters
(`preCycles`, `infCycles`, `postCycles`, `frameCycles`, `timingFrames`) can be read over J-Link
without resetting the core.

### Recovering a board that no longer boots the HP core

Symptom: after flashing, the screen stays dark; J-Link to `AE822FA0E5597LS0_M55_HP` fails with
"Could not read AHB ROM register / Could not find core", while `AE822FA0E5597LS0_M55_HE` still connects.
Cause (seen on 2026-10-17): an image that extended past 0x8057BC00 overwrote Alif's ATOC boot table
(entries DEVICE, A32_STUB, HP_APP, HE_STUB at the top of the HP partition), so the Secure Enclave stops
starting the HP core. `scripts/alif/flash_e8_hp.sh` now refuses such images.

Recovery that worked, using a full backup of the HP partition taken before flashing:

```bash
# backup (do this once, before experimenting): reads 0x80200000..0x80580000 (3.5 MB)
printf 'connect\nh\nsavebin backup.bin 0x80200000 0x380000\ng\nexit\n' > /tmp/b.jlink
JLinkExe -device AE822FA0E5597LS0_M55_HP -if SWD -speed 4000 -autoconnect 1 -CommanderScript /tmp/b.jlink

# restore THROUGH THE HE CORE (the HP core is unreachable), then power-cycle the board
printf 'connect\nh\nloadbin backup.bin 0x80200000\nexit\n' > /tmp/r.jlink
JLinkExe -device AE822FA0E5597LS0_M55_HE -if SWD -speed 4000 -autoconnect 1 -CommanderScript /tmp/r.jlink
```

Verify by reading 0x8057B000..0x80580000 back and comparing it with the same range of the backup.
Without a backup, rewrite the boot table with Alif SETOOLS (`app-gen-toc` + `app-write-mram`).
