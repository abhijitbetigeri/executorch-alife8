# BlazeFace for the ExecuTorch face-detection build

MediaPipe BlazeFace (front camera, 128×128) as an ExecuTorch program for the Ethos-U85-256 of the Alif Ensemble E8. Used by `alif_object_detection` when configured with `-DML_FRAMEWORK=ExecuTorch`.

| File | What it is |
|---|---|
| `blazeface_front_128_ethos-u85-256.pte` | The program the firmware embeds. ExecuTorch 1.1.0, int8, Vela `Ethos_U85_SRAM_MRAM` / `Shared_Sram` |
| `blazeface.py`, `blazeface.pth` | PyTorch port and weights by Matthijs Hollemans, https://github.com/hollance/BlazeFace-PyTorch (Apache-2.0, same terms as MediaPipe) |
| `export.py` | Re-creates the `.pte` |
| `calib/` | Put calibration frames here (png/jpg). Frames captured on the board's camera give the best int8 accuracy |

Model contract:
- Input 0: float NHWC 1×128×128×3, RGB in [0, 1]. The graph rescales to [-1, 1] and permutes to NCHW itself.
- Output 0: 1×896 score logits. Output 1: 1×896×16 box regressors (box centre, size, 6 keypoints).
- Decode, sigmoid threshold (0.75) and MediaPipe weighted NMS (IoU 0.3) run on the Cortex-M55 in `source/lib/mlek/use_case/object_detection/BlazeFaceProcessing.cc`. Anchors are generated there, not stored.

## Measured at export (ExecuTorch 1.1.0, Vela 4.5.0)

| | |
|---|---|
| NPU operators | 97 / 97 (100%); only `cortex_m::quantize/dequantize_per_tensor` at the boundary run on the CPU |
| `.pte` size | 189 KB |
| Vela SRAM / MRAM | 387 KB / 173 KB |
| MACs per frame | 31.7 M |

int8 matches float on reference photos (same detections). On-board latency: not measured yet.

## Re-exporting

The `.pte` must be written by the same ExecuTorch version as the runtime in `dependencies/executorch` (v1.1.0). Every pin below was needed on macOS (arm64):

```bash
uv venv --python 3.12 .venv-et110
uv pip install --python .venv-et110/bin/python "executorch==1.1.0" "torch==2.10.0" "ethos-u-vela==4.5.0" pillow
uv pip install --python .venv-et110/bin/python --no-deps "tosa-tools==2026.2.0" "ml_dtypes==0.5.1" "flatbuffers==24.3.25"
brew install flatbuffers          # ExecuTorch 1.1.0 calls `flatc` to write the .pte
.venv-et110/bin/python resources/object_detection/blazeface/export.py --calib 32
```

`set_up_default_resources.py --ml-framework=executorch` builds an equivalent environment on Linux.
