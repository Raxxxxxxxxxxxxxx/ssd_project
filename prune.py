"""
تقليم بنيوي (Structured Pruning) لنموذج A-SSD (القسم 4.3 من التقرير).

ملاحظة صادقة مهمة: `torch.nn.utils.prune.ln_structured` + `prune.remove()`
(كما ورد حرفياً في التقرير) يُصفّر أوزان القنوات الأقل أهمية لكنه **لا يغيّر
شكل الـTensor** - أي لا يقلّل حجم ملف الـcheckpoint ولا الـFLOPs الفعلية عند
حفظه بصيغة PyTorch العادية (القيم المُصفَّرة تبقى محفوظة كأصفار كثيفة). رقم
"11MB → 7.7MB" الوارد بالتقرير غير دقيق لنفس هذا الكود بدون خطوة إضافية
(إعادة بناء الطبقات فعلياً بعدد قنوات أقل، أو ضغط الملف). هذا السكربت يقيس
النتيجة الحقيقية بدل افتراض رقم التقرير، وينقل بوضوح الفائدة الفعلية لهذا
النوع من التقليم: تحديد القنوات الأقل أهمية (مفيد لإعادة هندسة يدوية للبنية
لاحقاً)، وليس تصغيراً فورياً للملف - ذلك يتحقق فعلياً عبر التكميم (quantize.py).

تحذير مهم آخر (نتيجة اختبار فعلي): نسبة 30% (amount=0.3) المذكورة بالتقرير
**تُدمِّر دقة النموذج بالكامل** على MobileNetV3-Small تحديداً (mAP@0.5 انخفض
من 38.78% إلى 0.00% فعلياً). السبب المرجّح: -16VGG (الأصلية بالتقرير) شبكة
مُفرطة المعاملات (over-parameterized) بتكرار كبير يتحمّل تقليماً بنسبة عالية،
بينما MobileNetV3-Small مصمَّمة أصلاً لتكون خفيفة بأقل تكرار ممكن - فتقليم
30% من كل طبقة على حدة عبر ~60 طبقة يتراكم بشكل مضاعف ويدمّر التمثيل
المتعلَّم بالكامل. الأنكى: حتى نسبة 10% (`--amount 0.1`) دمّرت الدقة بنفس
الشكل تقريباً (0.02% mAP) - أي أن المشكلة ليست فقط "30% كثيرة" بل أن
التقليم دفعة واحدة بدون أي إعادة تدريب (fine-tuning) غير قابل للتطبيق على
هذه المعمارية إطلاقاً بالطريقة الساذجة الموصوفة بالتقرير. التوصية الحقيقية:
تقليم تدريجي بنسب صغيرة جداً (1-5%) مع إعادة تدريب فعلية بعد كل خطوة
(Iterative Pruning + Fine-tuning)، وليس قصّاً لمرة واحدة كما هنا.

الاستخدام:
    python prune.py --checkpoint checkpoints_v2/last.pth --amount 0.3 \
        --output checkpoints_v2/last_pruned.pth
"""

import argparse
import os

import torch
import torch.nn.utils.prune as prune

from assd_model import AdaptiveSSD
from utils import VOC_CLASSES


def count_nonzero_params(model):
    total, nonzero = 0, 0
    for p in model.parameters():
        total += p.numel()
        nonzero += torch.count_nonzero(p).item()
    return total, nonzero


def prune_model(model, amount=0.3):
    """يقلّم كل طبقات Conv2d بنسبة `amount` من القنوات الأقل أهمية (L2 norm)
    حسب البعد dim=0 (قنوات الإخراج) - يطابق كود التقرير حرفياً."""
    pruned_layers = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, "weight", amount=amount, n=2, dim=0)
            prune.remove(module, "weight")
            pruned_layers += 1
    return pruned_layers


def main():
    parser = argparse.ArgumentParser(description="تقليم بنيوي لنموذج A-SSD")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--amount", type=float, default=0.3, help="نسبة القنوات المُقلَّمة (افتراضي 30%%)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_path = args.output or args.checkpoint.replace(".pth", "_pruned.pth")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = AdaptiveSSD(num_classes=len(VOC_CLASSES), anchor_wh=ckpt.get("anchor_wh"))
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    total_before, nonzero_before = count_nonzero_params(model)

    # نحفظ نسخة "قبل" بنفس مخطط الحفظ بالضبط (بدون optimizer state إن وُجد
    # بالـcheckpoint الأصلي) لضمان مقارنة عادلة - وإلا فإن أي فرق حجم قد يكون
    # سببه اختلاف المحتوى المحفوظ وليس التقليم نفسه.
    baseline_path = output_path.replace(".pth", "_baseline_unpruned.pth")
    torch.save({"model": model.state_dict(), "anchor_wh": ckpt.get("anchor_wh")}, baseline_path)
    size_before = os.path.getsize(baseline_path) / 1024 / 1024
    os.remove(baseline_path)

    pruned_layers = prune_model(model, amount=args.amount)

    total_after, nonzero_after = count_nonzero_params(model)

    torch.save({"model": model.state_dict(), "anchor_wh": ckpt.get("anchor_wh")}, output_path)
    size_after = os.path.getsize(output_path) / 1024 / 1024

    print(f"عدد طبقات Conv2d المُقلَّمة: {pruned_layers}")
    print(f"نسبة الأوزان غير الصفرية: قبل {nonzero_before/total_before*100:.1f}% "
          f"-> بعد {nonzero_after/total_after*100:.1f}%")
    print(f"حجم الملف: {size_before:.2f}MB -> {size_after:.2f}MB "
          f"({'لم يتغيّر فعلياً - كما هو متوقع، راجع الملاحظة أعلى الملف' if abs(size_before-size_after) < 0.5 else 'تغيّر'})")
    print(f"النسخة المُقلَّمة محفوظة في: {output_path}")
    print("\nلتقييم أثر التقليم على mAP فعلياً:")
    print(f"  python evaluate.py --data-root data/VOCdevkit/VOC2012 --checkpoint {output_path} --mode NORMAL")


if __name__ == "__main__":
    main()
