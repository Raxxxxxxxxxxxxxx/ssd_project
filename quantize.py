"""
تكميم INT8 بعد التدريب (Post-Training Static Quantization) لنموذج A-SSD
(القسم 4.3 من التقرير).

تصحيح مهم عن كود التقرير: التقرير يستخدم backend "fbgemm" وهو مُحسَّن
لمعالجات x86 (خوادم/حواسيب سطح مكتب) - بينما Raspberry Pi 4 معالجه ARM، وهو
ما يحتاج فعلياً backend "qnnpack" (مُحسَّن لـARM تحديداً). استخدام fbgemm هنا
كان سيُنتج نموذجاً "مُكمَّماً" لكن بدون أي تسريع فعلي عند تشغيله لاحقاً على
Raspberry Pi - أو قد لا يعمل إطلاقاً. هذا السكربت يستخدم qnnpack مباشرة.

ملاحظة صادقة مهمة (نتيجة تجربة فعلية وليست افتراضاً): بدون دمج (Fusion)
طبقات Conv+BatchNorm قبل التكميم، ينجح السكربت في التحويل والتشغيل الفعلي
(بعد حلّ عطلين حقيقيين واجهتهما: كتل Squeeze-and-Excitation وعمليات الجمع
المتبقي residual add تستخدمان عمليات Tensor غير مدعومة على أنواع مُكمَّمة،
تم إصلاحهما بـ DeQuant/Quant صريح وFloatFunctional على الترتيب) - لكن الحجم
الفعلي **لا يتغيّر تقريباً** (قِسناه: 4.24MB -> 4.24MB). السبب: 340 من أصل
536 Tensor بالنموذج بقيت float32 (أغلبها أوزان BatchNorm غير المدموجة)،
مقابل 44 فقط تحوّلت فعلياً لـint8. تطبيق Fusion يدوياً لكل كتل MobileNetV3
غير المُصمَّمة أصلاً للتكميم (torchvision لا توفّر قائمة دمج جاهزة لها) عمل
إضافي كبير لفائدة غير مؤكدة، بينما **مسار التكميم الحقيقي والأنسب لـ
Raspberry Pi هو INT8 عبر TFLite** (`export_tflite.py --quantize`) لأن
TFLite تتعامل مع هذه الأنماط المعمارية بنضج أكبر من التكميم الأصلي لـ
PyTorch، وهي المسار الذي سيُشغَّل فعلياً على الجهاز أصلاً. هذا السكربت
يبقى موجوداً كمرجع موثَّق للمحاولة ونتيجتها الحقيقية.

الاستخدام:
    python quantize.py --checkpoint checkpoints_v2/last.pth --mode NORMAL \
        --data-root data/VOCdevkit/VOC2012 --output checkpoints_v2/last_int8.pth
"""

import argparse
import os

import torch
import torch.ao.quantization as tq

from assd_model import AdaptiveSSD, MODE_CONFIG
from utils import VOC_CLASSES, preprocess


class QuantWrapper(torch.nn.Module):
    """يلفّ AdaptiveSSD بـ QuantStub/DeQuantStub اللازمين للتكميم الساكن،
    بوضع (mode) ثابت مطابقاً لنفس القيد الموجود في تصدير ONNX."""

    def __init__(self, model, mode):
        super().__init__()
        self.mode = mode
        self.quant = tq.QuantStub()
        self.model = model
        self.dequant_loc = tq.DeQuantStub()
        self.dequant_cls = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        loc, cls, _ = self.model(x, mode=self.mode)
        return self.dequant_loc(loc), self.dequant_cls(cls)


def count_size(path):
    return os.path.getsize(path) / 1024 / 1024


def main():
    parser = argparse.ArgumentParser(description="تكميم INT8 لنموذج A-SSD")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True, help="لبيانات المعايرة (calibration)")
    parser.add_argument("--mode", default="NORMAL", choices=["ECONOMY", "NORMAL", "TURBO"])
    parser.add_argument("--calib-images", type=int, default=32)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_path = args.output or args.checkpoint.replace(".pth", "_int8.pth")

    # qnnpack هو المحرك الصحيح لأجهزة ARM (Raspberry Pi 4/Jetson Nano)، وليس
    # fbgemm (x86) كما في كود التقرير - راجع الملاحظة أعلى الملف.
    torch.backends.quantized.engine = "qnnpack"

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = AdaptiveSSD(num_classes=len(VOC_CLASSES), anchor_wh=ckpt.get("anchor_wh"))
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    wrapped = QuantWrapper(model, args.mode)
    wrapped.eval()
    wrapped.qconfig = tq.get_default_qconfig("qnnpack")

    # كتل Squeeze-and-Excitation في MobileNetV3 (من torchvision) تستخدم ضرباً
    # مباشراً بين Tensor وSigmoid (scale * input) غير مدعوم على Tensors
    # مُكمَّمة (RuntimeError: empty_strided not supported on quantized
    # tensors). مجرد استثنائها بتعيين qconfig=None لا يكفي أيضاً: eager-mode
    # quantization (خلافاً لـFX mode) لا يُدرج تلقائياً حدود Quant/DeQuant
    # عند الانتقال بين منطقة مُكمَّمة ومنطقة float، فيصل Tensor مُكمَّم
    # (quint8) لطبقة float عادية وتفشل (خطأ bias mismatch). الحل: لفّ كل كتلة
    # SE بحدود DeQuant/Quant صريحة يدوياً.
    from torchvision.ops.misc import SqueezeExcitation

    class DequantizedSE(torch.nn.Module):
        def __init__(self, se_module):
            super().__init__()
            self.se = se_module
            self.dequant = tq.DeQuantStub()
            self.quant = tq.QuantStub()

        def forward(self, x):
            return self.quant(self.se(self.dequant(x)))

    def replace_se_blocks(module):
        count = 0
        for name, child in module.named_children():
            if isinstance(child, SqueezeExcitation):
                for submodule in child.modules():
                    submodule.qconfig = None
                setattr(module, name, DequantizedSE(child))
                count += 1
            else:
                count += replace_se_blocks(child)
        return count

    excluded = replace_se_blocks(wrapped)
    print(f"لفّ {excluded} كتلة Squeeze-and-Excitation بحدود DeQuant/Quant صريحة (تبقى float32 داخلياً).")

    # نفس المشكلة تتكرر مع الجمع المتبقي (residual add: result += input) في
    # كتل InvertedResidual داخل MobileNetV3 - عملية + عادية بين Tensors
    # مُكمَّمة غير مدعومة أيضاً (aten::add.out غير موجود لـQuantizedCPU).
    # الحل القياسي (المُستخدم فعلياً في torchvision.models.quantization):
    # استبدال + بـ nn.quantized.FloatFunctional().add()، وهذا لا يتطلب
    # استثناء الكتلة بالكامل من التكميم (خلافاً لـSE) لأن الجمع نفسه فقط هو
    # غير المدعوم - بقية طبقات Conv داخل الكتلة تبقى مُكمَّمة بشكل طبيعي.
    from torchvision.models.mobilenetv3 import InvertedResidual
    import torch.ao.nn.quantized as nnq

    def patch_residual_adds(module):
        count = 0
        for m in module.modules():
            if isinstance(m, InvertedResidual) and m.use_res_connect:
                m.add_module("skip_add", nnq.FloatFunctional())

                def new_forward(self, x):
                    return self.skip_add.add(self.block(x), x)

                m.forward = new_forward.__get__(m, InvertedResidual)
                count += 1
        return count

    patched = patch_residual_adds(wrapped)
    print(f"استبدال {patched} عملية جمع متبقٍ (residual add) بـ FloatFunctional متوافق مع التكميم.")

    tq.prepare(wrapped, inplace=True)

    # معايرة على عيّنة صور حقيقية من VOC2012 (وليست بيانات وهمية) حتى تعكس
    # نطاقات القيم الفعلية للتفعيلات - ضروري لدقة تكميم مقبولة.
    from train import VOCDataset

    dataset = VOCDataset(args.data_root, split="train")
    resolution = MODE_CONFIG[args.mode]["resolution"]

    with torch.no_grad():
        for i in range(min(args.calib_images, len(dataset))):
            frame, _, _ = dataset[i]
            tensor = preprocess(frame, resolution)
            wrapped(tensor)

    tq.convert(wrapped, inplace=True)
    wrapped.eval()

    torch.save({"model": wrapped.state_dict(), "anchor_wh": ckpt.get("anchor_wh"),
                "quantized": True, "mode": args.mode}, output_path)

    # مقارنة حجم عادلة (نفس المخطط بالضبط: state_dict + anchor_wh فقط)
    baseline_path = output_path + ".fp32_baseline_tmp"
    torch.save({"model": model.state_dict(), "anchor_wh": ckpt.get("anchor_wh")}, baseline_path)
    size_before = count_size(baseline_path)
    os.remove(baseline_path)
    size_after = count_size(output_path)

    print(f"حجم الملف (FP32 بنفس المخطط): {size_before:.2f}MB")
    print(f"حجم الملف (INT8): {size_after:.2f}MB  (تقليل {(1 - size_after/size_before) * 100:.1f}%)")
    print(f"النسخة المُكمَّمة محفوظة في: {output_path}")

    # التحقق الفعلي: تشغيل استدلال حقيقي بالنموذج المُكمَّم على صورة حقيقية
    frame, _, _ = dataset[0]
    tensor = preprocess(frame, resolution)
    with torch.no_grad():
        loc, cls = wrapped(tensor)
    print(f"\nالتحقق ناجح: النموذج المُكمَّم يعمل فعلياً - "
          f"loc={tuple(loc.shape)}, cls={tuple(cls.shape)}, "
          f"loc.dtype={loc.dtype} (dequantized إلى float عند الإخراج)")


if __name__ == "__main__":
    main()
