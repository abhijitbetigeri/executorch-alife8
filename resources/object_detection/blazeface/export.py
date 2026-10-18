# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Export MediaPipe BlazeFace (front, 128x128) to an ExecuTorch .pte for Ethos-U85-256 on Alif E8.

Run with ExecuTorch 1.1.0 (the runtime pinned in dependencies/executorch); see README.md.
Output: blazeface_front_128_ethos-u85-256.pte next to this script.
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
from blazeface import BlazeFace  # noqa: E402

from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner  # noqa: E402
from executorch.backends.arm.quantizer import (  # noqa: E402
    EthosUQuantizer,
    get_symmetric_quantization_config,
)
from executorch.backends.cortex_m.passes.quantized_op_fusion_pass import (  # noqa: E402
    QuantizedOpFusionPass,
)
from executorch.backends.cortex_m.passes.replace_quant_nodes_pass import (  # noqa: E402
    ReplaceQuantNodesPass,
)
from executorch.exir import (  # noqa: E402
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.dialects._ops import ops as exir_ops  # noqa: E402
from torch.fx.passes.operator_support import OperatorSupportBase  # noqa: E402
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e  # noqa: E402

SIZE = 128


class KeepTailOnCpu(OperatorSupportBase):
    """Keep BlazeFace's head tail (reshape / cat / squeeze) off the Ethos-U.

    With ExecuTorch 1.1.0 every `conv -> permute(0, 2, 3, 1) -> reshape` head lowers to
    CONV2D -> TRANSPOSE -> TRANSPOSE (the pair cancels) -> RESHAPE. Vela 4.5 (and 5.1) removes
    the transpose pair and then gives the conv the reshaped OFM shape (e.g. 1x1x1x512 instead of
    1x16x16x2), so the NPU computes the heads from the first pixels only and the scores come out
    scrambled. TOSA is correct (the TOSA reference model matches the int8 simulation bit for bit);
    only the Vela-compiled command stream is wrong. Running the cheap tail on the Cortex-M
    (portable view/cat/squeeze, ~15k int8 elements) avoids the pattern; outputs are unchanged.
    See experiments/fvp/FINDINGS.md.
    """

    _TAIL = {
        exir_ops.edge.aten.view_copy.default,
        exir_ops.edge.aten.cat.default,
        exir_ops.edge.aten.squeeze_copy.dims,
        exir_ops.edge.aten.squeeze_copy.dim,
        exir_ops.edge.aten.unsqueeze_copy.default,
    }

    def is_node_supported(self, submodules, node) -> bool:
        return node.target not in self._TAIL


class FaceNHWC(nn.Module):
    """Camera-layout input (1,128,128,3) float in [0,1] -> (scores 1x896, boxes 1x896x16)."""

    def __init__(self):
        super().__init__()
        self.net = BlazeFace()
        self.net.load_weights(os.path.join(HERE, "blazeface.pth"))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) * 2.0 - 1.0
        boxes, scores = self.net(x)
        return scores.squeeze(-1), boxes


def calibration_images(count: int) -> list:
    """Frames from calib/ (png/jpg captured on the board's camera), each in 4 rotations.

    Calibrate on real frames: with noise only, int8 BlazeFace scores every frame ~0.59 and
    finds no faces. The firmware runs the model on 4 rotations, so calibrate on them too.
    `count` is the number of source frames; noise only fills in when calib/ is empty.
    """
    from PIL import Image

    paths = sorted(glob.glob(os.path.join(HERE, "calib", "*.png")) +
                   glob.glob(os.path.join(HERE, "calib", "*.jpg")))
    imgs = []
    for path in paths[:count]:
        img = np.asarray(Image.open(path).convert("RGB").resize((SIZE, SIZE)))
        for quarter_turns in range(4):  # clockwise, as RotateRgb888Square on the device
            rotated = np.ascontiguousarray(np.rot90(img, -quarter_turns))
            imgs.append(torch.from_numpy(rotated).float().div(255).unsqueeze(0))
    if not imgs:
        print("WARNING: calib/ is empty; calibrating on noise gives a model that finds no faces")
        gen = torch.Generator().manual_seed(0)
        imgs = [torch.rand(1, SIZE, SIZE, 3, generator=gen) for _ in range(count)]
    print(f"calibration: {min(len(paths), count)} real frames x 4 rotations = {len(imgs)} samples")
    return imgs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calib", type=int, default=32, help="number of calibration frames")
    parser.add_argument("--out", default=os.path.join(HERE, "blazeface_front_128_ethos-u85-256.pte"))
    args = parser.parse_args()

    model = FaceNHWC().eval()
    calib = calibration_images(args.calib)
    example = (calib[0],)

    # Same configuration as docs/deploying_model_executorch_export.md.
    spec = EthosUCompileSpec(
        target="ethos-u85-256",
        config_ini=os.path.join(REPO, "scripts", "vela", "ensemble_vela.ini"),
        system_config="Ethos_U85_SRAM_MRAM",
        memory_mode="Shared_Sram",
        extra_flags=["--output-format=raw", "--debug-force-regor"],
    )

    start = time.time()
    graph = torch.export.export(model, example).module()
    quantizer = EthosUQuantizer(spec)
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
    prepared = prepare_pt2e(graph, quantizer)
    with torch.no_grad():
        for sample in calib:
            prepared(sample)
    quantized = convert_pt2e(prepared)

    edge = to_edge_transform_and_lower(
        torch.export.export(quantized, example),
        partitioner=[EthosUPartitioner(spec, additional_checks=[KeepTailOnCpu()])],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    edge = edge.transform([ReplaceQuantNodesPass(), QuantizedOpFusionPass()])
    program = edge.to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False))
    with open(args.out, "wb") as f:
        f.write(program.buffer)

    cpu_ops = sorted({m.decode() for m in re.findall(rb"(?:aten|quantized_decomposed|cortex_m)::\w+",
                                                      program.buffer)})
    print(f"wrote {args.out}: {len(program.buffer) / 1024:.1f} KB in {time.time() - start:.0f} s; "
          f"CPU ops: {cpu_ops}")


if __name__ == "__main__":
    main()
