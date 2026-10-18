"""MobileNetV2 (torchvision IMAGENET1K_V2) at 160x160 for the MLEK alif_img_class ExecuTorch app.

Input: float NCHW 1x3x160x160, ImageNet-normalised (the app applies mean/std itself).
Output: 1x1000 logits, a plain vector (no detection-head reshapes).
"""
import glob, os, re, sys
import numpy as np, torch
from PIL import Image
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config
from executorch.backends.cortex_m.passes.quantized_op_fusion_pass import QuantizedOpFusionPass
from executorch.backends.cortex_m.passes.replace_quant_nodes_pass import ReplaceQuantNodesPass
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

SIZE = 160
REPO = os.path.expanduser("~/projects/arm-executorch-hack/executorch-alife8")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
weights = MobileNet_V2_Weights.IMAGENET1K_V2
labels_all = weights.meta["categories"]
# Everyday desk/room objects you can point the camera at. Keeping a subset of the ImageNet head
# shrinks the program from 3.4 MB to ~2.3 MB, under the board's 3.5 MB M55-HP MRAM partition.
KEEP = ["computer keyboard", "mouse", "laptop", "notebook", "desktop computer", "monitor", "screen",
        "television", "cellular telephone", "remote control", "water bottle", "water jug", "coffee mug",
        "cup", "beer glass", "wine bottle", "pop bottle", "backpack", "purse", "wallet", "sunglasses",
        "sunglass", "ballpoint", "fountain pen", "pencil box", "rubber eraser", "paper towel",
        "toilet tissue", "envelope", "binder", "book jacket", "comic book", "menu", "digital clock",
        "wall clock", "analog clock", "lampshade", "table lamp", "desk", "folding chair", "studio couch",
        "pillow", "banana", "orange", "Granny Smith", "plate", "soup bowl", "sock", "running shoe",
        "jersey", "sweatshirt", "suit", "hard disc", "modem", "switch", "screwdriver", "loudspeaker",
        "microphone", "joystick", "space bar", "printer", "projector", "electric fan", "iPod",
        "hand-held computer", "lab coat", "cardigan", "spectacles"]
keep_idx = [labels_all.index(n) for n in KEEP if n in labels_all]
missing = [n for n in KEEP if n not in labels_all]
if missing: print("not in ImageNet labels:", missing)
labels = [labels_all[i] for i in keep_idx]

def load(p):
    im = Image.open(p).convert("RGB"); w, h = im.size; s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((SIZE, SIZE))
    return torch.from_numpy(np.asarray(im).copy()).float().div(255).unsqueeze(0)  # NHWC [0,1]

paths = sorted(glob.glob("calib/*"))
calib = [load(p) for p in paths]
class NpuFriendlyMV2(torch.nn.Module):
    """torchvision MobileNetV2 with the adaptive pool + flatten expressed as a fixed
    average pool and a reshape, so the whole graph lowers to the Ethos-U (no CPU kernels)."""
    def __init__(self):
        super().__init__()
        base = mobilenet_v2(weights=weights).eval()
        self.features = base.features
        full = base.classifier[1]
        head = torch.nn.Linear(full.in_features, len(keep_idx))
        with torch.no_grad():
            head.weight.copy_(full.weight[keep_idx]); head.bias.copy_(full.bias[keep_idx])
        self.classifier = torch.nn.Sequential(base.classifier[0], head)
        self.pool = torch.nn.AvgPool2d(SIZE // 32)
    def forward(self, x):
        # Camera layout in: NHWC float RGB in [0, 1]. ImageNet normalisation and the NCHW
        # permute are folded in here, so the firmware only converts bytes to float.
        x = (x.permute(0, 3, 1, 2) - MEAN) / STD
        x = self.pool(self.features(x))
        return self.classifier(x.reshape(1, 1280))

model = NpuFriendlyMV2().eval()
spec = EthosUCompileSpec(target="ethos-u85-256", config_ini=f"{REPO}/scripts/vela/ensemble_vela.ini",
                         system_config="Ethos_U85_SRAM_MRAM", memory_mode="Shared_Sram",
                         extra_flags=["--output-format=raw", "--debug-force-regor"])
graph = torch.export.export(model, (calib[0],)).module()
q = EthosUQuantizer(spec); q.set_global(get_symmetric_quantization_config(is_per_channel=True))
prep = prepare_pt2e(graph, q)
with torch.no_grad():
    for c in calib: prep(c)
quant = convert_pt2e(prep)
with torch.no_grad():
    for p, c in zip(paths, calib):
        f = model(c)[0]; qv = quant(c)[0]
        print(f"{os.path.basename(p):14s} float: {labels[f.argmax()]:<22s} int8 sim: {labels[qv.argmax()]}")
edge = to_edge_transform_and_lower(torch.export.export(quant, (calib[0],)), partitioner=[EthosUPartitioner(spec)],
                                   compile_config=EdgeCompileConfig(_check_ir_validity=False))
edge = edge.transform([ReplaceQuantNodesPass(), QuantizedOpFusionPass()])
prog = edge.to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False))
out = f"mv2nhwc_160_desk{len(labels)}_ethos-u85-256.pte"; open(out, "wb").write(prog.buffer)
open(f"labels_desk{len(labels)}.txt", "w").write("\n".join(labels) + "\n")
print("wrote", out, f"{len(prog.buffer)/1024:.0f} KB; CPU ops:", sorted({m.decode() for m in re.findall(rb'(?:aten|quantized_decomposed|cortex_m)::\w+', prog.buffer)}))
