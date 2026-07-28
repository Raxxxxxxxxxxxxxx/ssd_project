"""
تصدير A-SSD إلى ONNX (القسم 4.4 من التقرير) — الخطوة الأولى قبل النشر على
Jetson Nano (عبر TensorRT) أو أي محرك يدعم ONNX. يُصدَّر النموذج بوضع ودقة
ثابتين (mode/resolution) لأن الشكل الديناميكي متعدد الأوضاع لا يمكن تصديره
كرسم بياني (graph) واحد ثابت.

الاستخدام:
    python export_onnx.py --checkpoint checkpoints_v2/last.pth --mode NORMAL \
        --output assd_normal.onnx
"""

import argparse

import numpy as np
import torch

from assd_model import AdaptiveSSD, MODE_CONFIG
from utils import VOC_CLASSES


class ONNXWrapper(torch.nn.Module):
    """يلفّ AdaptiveSSD بوضع/دقة ثابتين لأن forward() الأصلي يتفرّع حسب mode
    (نص)، وONNX لا يدعم تصدير هذا التفرّع الديناميكي مباشرة."""

    def __init__(self, model, mode):
        super().__init__()
        self.model = model
        self.mode = mode

    def forward(self, x):
        loc, cls, _ = self.model(x, mode=self.mode)
        return loc, cls


def main():
    parser = argparse.ArgumentParser(description="تصدير A-SSD إلى ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", default="NORMAL", choices=["ECONOMY", "NORMAL", "TURBO"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_path = args.output or f"assd_{args.mode.lower()}.onnx"

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = AdaptiveSSD(num_classes=len(VOC_CLASSES), anchor_wh=ckpt.get("anchor_wh"))
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    wrapped = ONNXWrapper(model, args.mode).eval()
    resolution = MODE_CONFIG[args.mode]["resolution"]
    dummy_input = torch.randn(1, 3, resolution[1], resolution[0])

    torch.onnx.export(
        wrapped, dummy_input, output_path,
        input_names=["input"], output_names=["loc", "cls"],
        opset_version=18,
    )
    print(f"تم التصدير إلى {output_path} (وضع {args.mode}, دقة {resolution})")

    # التحقق الفعلي: تشغيل الملف المُصدَّر عبر onnxruntime ومقارنته بمخرجات
    # PyTorch الأصلية على نفس المدخل - وليس فقط افتراض أن التصدير نجح.
    import onnxruntime as ort

    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    onnx_loc, onnx_cls = session.run(None, {"input": dummy_input.numpy()})

    with torch.no_grad():
        torch_loc, torch_cls = wrapped(dummy_input)

    loc_diff = np.abs(onnx_loc - torch_loc.numpy()).max()
    cls_diff = np.abs(onnx_cls - torch_cls.numpy()).max()
    print(f"أقصى فرق مع PyTorch: loc={loc_diff:.6f}, cls={cls_diff:.6f}")

    assert loc_diff < 1e-3 and cls_diff < 1e-3, "فرق كبير بين ONNX وPyTorch - التصدير غير صحيح!"
    print("التحقق ناجح: مخرجات ONNX مطابقة لمخرجات PyTorch.")


if __name__ == "__main__":
    main()
