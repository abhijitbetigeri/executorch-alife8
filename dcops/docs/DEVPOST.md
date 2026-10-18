# DC-Ops on E8: an air-gapped data-center component identifier

> Draft Devpost text. Every `TODO(measure)` is a number we have not measured yet. Fill it from
> `dcops/REPORT.md`, `dcops/models/hub/*/metrics.json` / `export.json`, or the board, and do not guess.
> Repo: `TODO(link)` · Judged commit/tag: `TODO(tag)`

## Inspiration

Secure data centers often ban phones, cameras and cloud services on the floor. Technicians still
have to identify components they rarely touch: a power shelf, a drive bay, a management port, an
amber LED. They do it from memory or a printed binder. We wanted a small device with one fixed
function that runs fully offline: point it at the hardware and it tells you what the part is and
what its LEDs mean, with nothing to upload and no general-purpose camera app.

**Prior work, disclosed:** this ports our earlier phone project
[DC-Ops](https://github.com/abhijitbetigeri/DC-Ops) (Qualcomm x Meta ExecuTorch hackathon:
YOLOv8n-seg on a Snapdragon NPU, 319-image auto-labelled dataset, 16 component classes, component
info cards). We reused its dataset, its fine-tuned YOLOv8n-seg detector, and its info-card text
(`dcops/dc_components.json`). Everything built at this event is new: the distillation pipeline that
turns the phone detector into an MCU-sized classifier, the ExecuTorch/Ethos-U85 export, the E8
firmware, and the keyword-spotting voice control.

## What it does

- The MT9M114 camera streams a live preview to the 480x800 LCD.
- Each frame, a MobileNetV2 classifier on the **Ethos-U85** labels the centre of the view as one
  of 16 data-center components or background. The top-2 labels and confidences are drawn over the
  preview.
- An info card from the DC-Ops database shows the component's description, specs and LED meanings.
  `TODO(confirm)`: whether the card is on the LCD in the judged build.
- Voice keywords (`go`, `stop`, `yes`, `no`, `up`, `down`, ...) from the board's microphone drive the
  workflow without touching the device. `TODO(confirm)`: the keyword-to-action mapping in the judged build.
- No network, no cloud, no storage of images.

## How we built it

**Distillation pipeline (GPU, `dcops/jobs/`)**

1. `teacher.py`: the DC-Ops YOLOv8n-seg detector runs over the 319 DC-Ops images. Its boxes,
   merged with the dataset's own unambiguous labels, become square component crops (224 px,
   1.3x context), plus background crops that avoid every box. That gives **3,494 crops (3,076 train /
   418 val, split by source image)**. A ConvNeXt-Small (ImageNet-22k) teacher is fine-tuned on them
   with class-balanced sampling and writes soft labels (flip-TTA logits).
   Teacher val acc: `TODO(measure)`.
2. `student.py`: a MobileNetV2 (ImageNet init) is trained with knowledge distillation (T=4, alpha=0.7)
   plus a **board-camera augmentation**: resolution loss, blur, bluish colour cast, gamma/contrast
   wash-out, sensor noise, glare, and screen moire, because demo images are often shown on a laptop.
   The model takes the camera layout directly (NHWC float RGB [0,1]), has ImageNet normalisation
   folded in, and ends in a fixed AvgPool + reshape so the whole graph lowers to the NPU.
   Runs: 128/160/192 px, a 120-epoch run, and two ablations (no KD, no board-camera aug).
3. `kws.py`: a DS-CNN keyword spotter distilled from `MIT/ast-finetuned-speech-commands-v2`
   (Audio Spectrogram Transformer, ~87M params) on Speech Commands v0.02. The model is trained on
   a torch re-implementation of the **exact MFCC the firmware computes** (MLEK `Mfcc.cc`,
   MicroNetKwsMFCC: 16 kHz, 640/320 Hann frames, 1024-pt FFT, 40 HTK mel bands 20–4000 Hz, 10 DCT
   coefficients, 49x10 per second). The script checks it against a NumPy port, so there is no
   train/deploy feature skew. The 12 labels are in the order of `resources/kws/labels`.
4. `dcops/export/`: int8 PT2E quantization with the Ethos-U quantizer, calibrated on real validation
   crops (vision) or board-identical MFCCs (KWS). The models are lowered with `EthosUPartitioner`
   for `ethos-u85-256` (ExecuTorch 1.1) and written as a `.pte`. Each export reports the CPU-side
   ops left over and the float/int8 agreement.

Training ran overnight in parallel on HF Jobs / Colab GPUs (`dcops/launch_overnight.sh`,
`dcops/colab/dcops_overnight.ipynb`).

**Firmware (`source/app/use_case/alif_object_detection`, `-DML_FRAMEWORK=ExecuTorch -Dalif_object_detection_ET_TASK=CLASSIFY`)**

We started from the Alif ML Embedded Evaluation Kit and a teammate's DevKit-E8 camera pipeline
(MT9M114 MIPI -> CPI/DMA -> rotate/crop/resize -> LVGL preview). We then added an ExecuTorch
runtime path with the Ethos-U85 delegate. On it we first brought up BlazeFace face detection and then
a generic classifier task: RGB888 -> float, one NPU inference, softmax top-3 on the M55, and the
label drawn on the LCD.

## Architecture and workload placement

```
MT9M114 --MIPI/CPI+DMA--> SRAM frame --M55 HP: rotate, crop, resize--> 160x160 RGB888
   --M55 HP: ->float NHWC--> [int8 quantize, M55] --Ethos-U85-256: MobileNetV2 (all ops)--> [dequantize, M55]
   --M55 HP: softmax top-3 + info-card lookup--> LVGL on CDC200/MIPI-DSI --> ILI9806E LCD
Mic --PDM/I2S--> M55: MFCC 49x10 --NPU: DS-CNN--> keyword --> UI state machine
```

| Workload | Runs on | Notes |
|---|---|---|
| Camera capture | MIPI CSI + CPI + DMA into SRAM | MT9M114, 1280x720 RGB565 |
| Preprocess (rotate 180°, full-screen preview, centre-crop + resize to model input) | Cortex-M55 HP | camera HAL; dominates the frame time |
| Input conversion RGB888 -> float NHWC | Cortex-M55 HP | normalisation folded into the model |
| int8 quantize / dequantize at graph edges | Cortex-M55 HP | ExecuTorch `cortex_m` kernels |
| Vision classifier (MobileNetV2) | **Ethos-U85-256** | all ops delegated (64–67 of 64–67 ops for the 160 px build we checked) |
| Postprocess (softmax, top-3) | Cortex-M55 HP | trivial cost |
| UI: preview, label, info card | M55 HP + CDC200 display controller + MIPI DSI | LVGL |
| Audio capture | PDM/I2S mic + DMA | `TODO(confirm)` in judged build |
| MFCC (49x10) | Cortex-M55 `TODO(confirm HP or HE)` | identical to training features |
| Keyword spotter (DS-CNN) | `TODO(confirm)`: Ethos-U85 (exported for `ethos-u85-256`) | HE + Ethos-U55 is the low-power alternative, not built |

Why this split: the NPU takes the dense convolutions. The M55 keeps the work that does not map to
it, like camera geometry, softmax and UI, plus the int8 boundary. We kept the classifier small
(MobileNetV2, trimmed head) rather than port the YOLOv8n-seg. It fits the 3.5 MB HP MRAM partition
and keeps NPU time far below camera time.

## Performance

Measured on the DevKit-E8, M55 HP at 400 MHz, Release build, GCC 14.3.1. Times come from the
firmware's cycle counters (`preCycles`, `infCycles`, `postCycles`, `frameCycles`), read over J-Link.

| Metric | Value | Status |
|---|---|---|
| BlazeFace 128 px, NPU inference incl. int8 boundary | 4.61 ms | measured (26-frame average) |
| BlazeFace pre / post | 1.08 ms / 0.05 ms | measured |
| Frame-to-frame (camera + preview + model + LVGL) | 166 ms, 6.0 FPS | measured (BlazeFace build) |
| Classifier build frame rate | ~6 FPS | observed on board |
| Classifier (MobileNetV2 160 px) NPU inference | `TODO(measure)` ms | read `infCycles` |
| Classifier pre / post | `TODO(measure)` ms | |
| Classifier .pte (67-class ImageNet subset, validation build) | 2.35 MB, all ops on the U85 | measured |
| DC-Ops student .pte (17 classes) | `TODO(measure)` KB, CPU ops: `TODO` | `export.json` |
| DC-Ops student val acc / board-sim acc | `TODO(measure)` / `TODO(measure)` | `metrics.json` |
| Ablation: no KD / no board-camera aug | `TODO(measure)` / `TODO(measure)` | |
| KWS test acc / params / .pte size | `TODO(measure)` | `kws/*/metrics.json` |
| KWS latency (MFCC + inference) | `TODO(measure)` | |
| MRAM used (HP image, validation build) | 3.16 MiB of 3.47 MiB usable, 312 KB headroom | from ELF |
| SRAM used | 6.86 MiB bulk SRAM (frames, preview, 1.5 MiB ExecuTorch arena) + 750 KiB framebuffer | from ELF |
| Power | `TODO(measure)` mW | |

**End-to-end check:** on the same camera frame, the board's top-3 (computer keyboard 12.6%,
rubber eraser 9.0%, television 8.7%) matched our PC int8 simulation (13.0%, 9.0%, 8.1%). That
confirms the camera -> preprocessing -> NPU -> postprocessing path is bit-faithful before the DC-Ops
model is swapped in.

## Challenges we ran into

- **We bricked HP boot, then recovered it.** The HP app partition is 0x80200000–0x80580000, and
  Alif's ATOC boot table sits at its top (0x8057BC00). The linker script lets an oversized image link,
  and flashing that image overwrote the boot table, so the HP core stopped booting. We restored an
  MRAM backup through the HE core. `scripts/alif/flash_e8_hp.sh` now refuses any image that ends
  past 0x80578000. The same limit is why we trimmed the classifier head, which cut the .pte from
  3.4 MB to 2.35 MB.
- **The camera hangs after a J-Link reset.** The MT9M114 does not come back after a debugger
  reset, so every flash ends with a power-cycle.
- **Output-layout bug in BlazeFace.** The int8 program's output layout did not match the reference
  decoder. We found it by comparing on-host decode against PyTorch.
- **Domain gap.** The DC-Ops images are clean web photos, while the board camera is low-res,
  bluish and washed out, and is often pointed at a laptop screen. We modelled that gap in the
  augmentation instead of hoping it away.
- **Class imbalance.** The teacher crops are uneven. For example, drive bay has 759 crops, network
  port 702 and compute tray 351, but power connector has 5. Four classes have no crops at all: NVLink
  switch tray, network switch, cable cartridge and management port. Those four outputs are
  untrained, and we say so.
- **Console.** The default UART2 is not on the PRG USB bridge, so we read counters over J-Link.

## Accomplishments we're proud of

- A full ExecuTorch -> Ethos-U85 path on the E8 with 100% NPU delegation for both BlazeFace
  (97/97 ops) and MobileNetV2, measured on hardware.
- A board result that agrees with the PC int8 simulation to within about 0.6 percentage points
  per class.
- A distillation pipeline from a phone detector (YOLOv8n-seg) to an MCU classifier, with ablations
  and a camera-domain model of the real board.
- A KWS trained on the board's exact MFCC, so the features at deploy time are the features it
  was trained on.
- A flash guard that turns a brick into a refused command.

## What we learned

- On this board the NPU is not the bottleneck. BlazeFace inference is 3.4% of the frame, and the
  camera, preview resize and LVGL on the M55 take the rest. The next speedup is in the pixel
  pipeline (smaller preview, the ISP or GPU2D for resize), not the model.
- MRAM partitions and boot tables matter more than total MRAM size.
- When train and deploy features match exactly (MFCC, NHWC [0,1] input), there are fewer
  silent accuracy losses.

## What's next

- Move the KWS to the M55 HE + Ethos-U55 so the always-on voice path runs on the low-power core,
  and use a keyword to wake the HP core.
- Collect real board-camera images in a lab rack and fine-tune on them.
- Add detection (multiple components per frame), and fill the empty classes with more data.
- Use the ISP/GPU2D for preview scaling to lift the frame rate.
- Program the image with SETOOLS so the device boots standalone without a debugger.

## Built with

ExecuTorch 1.1 · Arm Ethos-U85 / Vela · Arm ML Embedded Evaluation Kit · Alif Ensemble E8 DevKit ·
PyTorch · timm (ConvNeXt) · torchvision (MobileNetV2) · Ultralytics YOLOv8 · Hugging Face
Transformers (AST) · LVGL · GCC 14.3.1 · SEGGER J-Link · Hugging Face Jobs / Colab

## Team

`TODO(names)`. Camera pipeline: Nissan Pow.
