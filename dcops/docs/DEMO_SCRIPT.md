# DC-Ops on E8: live demo run-of-show (2–3 min)

## Before judges arrive (T-10 min)

- [ ] Flash the judged build with `scripts/alif/flash_e8_hp.sh`, then **power-cycle** the board
      (unplug PRG USB for a few seconds). The camera hangs after a J-Link reset.
- [ ] Preview is live on the LCD and labels update.
- [ ] Laptop: full-screen photo slideshow of the strongest classes (below), brightness at max,
      glare-free angle. A second tab holds the perf table and the architecture diagram (`ARCHITECTURE.md`).
- [ ] Printed photos or real hardware (a drive caddy, a fan, a NIC with ports) as backup props.
- [ ] Laptop with J-Link connected, the counter-read command ready (see "Perf numbers").
- [ ] Phone timer. Quiet spot for the voice part, or a plan to skip it.
- [ ] A backup video of the working demo on the laptop: `TODO(record)`.

## Strongest classes (most training crops)

drive bay (759) · network port (702) · compute tray (351) · power shelf (245) · LED indicator (148) ·
fan (77). Lead with these. Avoid NVLink switch tray, network switch, cable cartridge and management
port: they had **no** training crops. Also avoid power connector (5 crops).

## Script

| Time | Do | Say |
|---|---|---|
| 0:00 | Hold up the board | "Secure data centers ban phones and cloud. This is a single-purpose, offline identifier on an Alif E8. Camera in, answer on screen, nothing leaves the device." |
| 0:15 | Point at a drive-bay photo | "The Ethos-U85 runs a MobileNetV2 on every frame. Drive bay, plus its info card." |
| 0:35 | Network ports, then power shelf, then a fan or LED photo | "Each label is a full NPU inference. The card shows the specs and what the LED colours mean." |
| 1:00 | Say "go" / "stop" / "up" / "down" (`TODO(confirm mapping)`) | "Hands are busy in a rack, so a keyword spotter on the same chip drives the UI: MFCC on the M55, DS-CNN on the NPU." |
| 1:25 | Show the perf slide / J-Link readout | "NPU inference is X ms (`TODO(measure)`). The whole frame is about 166 ms at 6 FPS, and the NPU is only about 3% of that. The camera and preview on the M55 are the bottleneck, which is exactly where we would optimise next." |
| 1:50 | Show the distillation diagram | "We reused our DC-Ops phone detector as a teacher: YOLO crops, a ConvNeXt teacher, then a MobileNetV2 student with board-camera augmentation. We validated the board against a PC int8 simulation to within 0.6 points." |
| 2:15 | Show the memory map | "The 3.5 MB HP partition is a hard limit. We bricked boot once, recovered through the HE core, and wrote a flash guard." |
| 2:35 | Close | "Fully offline, NPU-accelerated, voice-driven. Questions?" |

## Fallbacks

| Problem | Do this |
|---|---|
| Camera frozen or black | Power-cycle (unplug PRG USB, about 5 s). Don't reset with J-Link. If it is still dead, play the backup video and walk through the perf table. |
| Wrong label on a real object | Switch to a laptop photo of a strong class. Training includes screen moire and glare. Fill the frame with the component and hold it 20–30 cm away. |
| Labels flicker | Hold steady for 1 s and point at the centre of the frame (the model sees a centre crop). |
| Laptop glare | Tilt the screen, lower brightness or use a printed photo. |
| Voice doesn't trigger | Speak close to the mic, one word, in a quiet moment. If the room is too loud, say "it's noisy here" and show the KWS accuracy from `REPORT.md`. |
| DC-Ops model misbehaves | Flash the validated 67-class ImageNet build (`build-classify`) and show the board-vs-PC match: keyboard 12.6/13.0%, eraser 9.0/9.0%, TV 8.7/8.1%. Then **power-cycle**. |
| Board won't boot after flash | Don't debug live. Use the second board / backup video. Recovery means writing the MRAM backup (`../board-backups/`) through the HE core, which is too slow for a demo slot. |

## Showing perf numbers

- Preferred: a printed or on-screen table from `DEVPOST.md` / `REPORT.md` with measured values.
- Live: the console is on UART2 by default, which is not on the PRG USB bridge. Either build with
  `-DCONSOLE_UART=4` and set SW4 to UART4, or read the counters with J-Link:
  `JLinkExe -device AE822FA0E5597LS0_M55_HP -if SWD -speed 4000`, then `connect`, then `mem32`
  on `infCycles`, `preCycles`, `postCycles`, `frameCycles` and `timingFrames` (addresses in
  `build-*/bin/*.map`). ms = cycles / frames / 400,000. Reading memory does not reset the core, so
  the camera keeps running.
- Optional: `-Dalif_object_detection_SHOW_INF_TIME=ON` draws the inference time on screen
  (`TODO(confirm)` for the classify path).
