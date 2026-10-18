# DC-Ops on E8: judging checklist (Sat Oct 18, deadline 5:00 pm PDT)

The owners are placeholders. Assign them at the morning stand-up. The times are estimates.

| Criterion (pts) | Evidence we have | Still needed | Owner | Est. |
|---|---|---|---|---|
| Functionality & reliability (20) | Classifier path runs on the board at about 6 FPS. Board top-3 matches the PC int8 sim. BlazeFace runs on the board. A flash guard prevents the boot brick. | DC-Ops student flashed and identifying real/laptop images. A 10-minute soak run without hangs. A backup video. A power-cycle step in the demo. | firmware | 2–3 h |
| Performance & efficiency (20) | BlazeFace: 4.61 ms NPU, 1.08 ms pre, 0.05 ms post, 166 ms/frame (6.0 FPS). MRAM/SRAM from the ELF. | Classifier NPU/pre/post ms for the DC-Ops build. KWS latency. **Power** (mW, idle vs inference: the `alif_object_detection_power` use case or a USB meter). One before/after optimisation (e.g. 128 vs 160 px, or smaller preview). | perf | 2 h |
| Effective use of E8 hardware (15) | U85 runs 100% of ops for both models. Camera DMA, CDC200/DSI and LVGL in use. Memory map and partition analysis done. Placement table written. | Confirm the KWS placement. Optional: KWS on HE + U55 (high risk, skip unless the rest is done). A one-line rationale per placement in the README. | firmware/voice | 0.5 h docs; HE port 4 h+ |
| Multimodal & sensor integration (15) | Camera + LCD working. KWS trained on the board's exact MFCC. | **Mic -> MFCC -> KWS running on the board and driving the UI.** This is the biggest open item. Keyword-to-action mapping documented. | voice | 3–4 h |
| Innovation & use-case value (10) | Air-gapped DC use case, info cards from DC-Ops, a clear "why a microcontroller" story. | Info card on the LCD (not just a label). One sentence on deployment constraints in the pitch. | firmware-ui | 2 h |
| Technical ambition (10) | YOLO -> ConvNeXt -> MobileNetV2 distillation. Board-camera domain augmentation. AST -> DS-CNN distillation. Two models. Ablations queued. | Ablation numbers in `REPORT.md` (KD vs no KD, aug vs no aug). Teacher accuracy. | ML | 0.5 h (if training finished) |
| Engineering quality & demo (10) | Docs in `dcops/docs/`, guard script, reproducible jobs and export scripts. | Public repo, commit tag, README section merged, a rehearsed 2–3 min demo, prior-work disclosure in the README. | all | 1.5 h |

## Submission form must-haves

- [ ] Public repo URL. `TODO`: confirm that nothing private, such as HF tokens or the private Hub repo
      names, is required to build. Note that the student/crops Hub repos are private: make them public
      or ship the `.pte` in the repo.
- [ ] Commit hash / tag of the judged build (`git tag dcops-e8-judged`), done by a human, after review.
- [ ] README with build / export / flash / run (`README_SECTION.md`).
- [ ] Architecture and workload placement (`ARCHITECTURE.md`, `DEVPOST.md`).
- [ ] Perf: latency, memory, power. Replace every `TODO(measure)`.
- [ ] Disclosure of reuse from github.com/abhijitbetigeri/DC-Ops.
- [ ] Demo video (optional but a strong fallback).

## Suggested timeline (Oct 18)

| Time | Milestone |
|---|---|
| 08:00 | Read `REPORT.md`, pick the student (160 px KD unless board-sim acc says otherwise), export and check the .pte fits under 0x80578000 |
| 09:30 | DC-Ops build flashed, power-cycled, sanity check on laptop photos |
| 11:00 | KWS on board, or a decision to cut it (call it at noon at the latest) |
| 12:00 | Perf and power measured, numbers filled in |
| 13:30 | Info card UI final; record the backup video |
| 15:00 | Docs/README merged, tag created, Devpost draft complete |
| 16:00 | Rehearse the demo twice; submit by **16:30** (30 min buffer) |
