"""
حساب mAP@0.5 حقيقي على مجموعة تحقق (val) بأسلوب Pascal VOC القياسي،
لمقارنة نتائج A-SSD المدرَّب فعلياً بالرقم المذكور في التقرير (%71.4).

الاستخدام:
    python evaluate.py --data-root /path/to/VOCdevkit/VOC2012 \
                        --checkpoint checkpoints/last.pth --mode NORMAL
"""

import argparse
from collections import defaultdict

import torch
from torchvision.ops import box_iou

from assd_model import AdaptiveSSD
from train import VOCDataset
from utils import VOC_CLASSES, preprocess, postprocess


def compute_ap(recalls, precisions):
    """AP بأسلوب الاستيفاء الكامل (all-point interpolation) المعتمد في VOC2012."""
    recalls = [0.0] + list(recalls) + [1.0]
    precisions = [0.0] + list(precisions) + [0.0]

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return ap


@torch.no_grad()
def evaluate(model, dataset, mode, conf_thresh=0.01, nms_thresh=0.45, iou_thresh=0.5, device="cpu"):
    model.eval()

    # لكل صنف: قائمة (confidence, is_true_positive) ولكل صورة: صناديق الحقيقة المتبقية غير المطابَقة
    all_predictions = defaultdict(list)
    all_gts_by_class = defaultdict(dict)  # class_id -> {image_idx: {"boxes":..., "matched":[bool,...]}}
    num_gt_per_class = defaultdict(int)

    for idx in range(len(dataset)):
        frame, gt_boxes, gt_labels = dataset[idx]
        h, w = frame.shape[:2]

        for box, label in zip(gt_boxes.tolist(), gt_labels.tolist()):
            class_id = label + 1  # +1 لأن 0 محجوزة للخلفية في مخرجات النموذج
            # تحويل صندوق الحقيقة من الإحداثيات المُطبَّعة [0,1] إلى بكسل، لأن
            # postprocess() يُخرج صناديق التوقع بالبكسل مباشرة - أي مقارنة IoU
            # بين نظامي إحداثيات مختلفين تُعطي صفراً دائماً بغض النظر عن جودة النموذج.
            xmin, ymin, xmax, ymax = box
            pixel_box = [xmin * w, ymin * h, xmax * w, ymax * h]
            entry = all_gts_by_class[class_id].setdefault(idx, {"boxes": [], "matched": []})
            entry["boxes"].append(pixel_box)
            entry["matched"].append(False)
            num_gt_per_class[class_id] += 1

        from assd_model import MODE_CONFIG
        resolution = MODE_CONFIG[mode]["resolution"]
        input_tensor = preprocess(frame, resolution).to(device)
        loc_preds, cls_preds, anchors = model(input_tensor, mode=mode)
        detections = postprocess(
            loc_preds, cls_preds, anchors, orig_size=(w, h),
            conf_thresh=conf_thresh, nms_thresh=nms_thresh, top_k=200,
        )

        for box, class_id, score in detections:
            all_predictions[class_id].append((score, idx, box))

        if (idx + 1) % 50 == 0:
            print(f"  ... تمت معالجة {idx + 1}/{len(dataset)} صورة")

    aps = {}
    for class_id in range(1, len(VOC_CLASSES)):
        preds = sorted(all_predictions[class_id], key=lambda p: p[0], reverse=True)
        n_gt = num_gt_per_class[class_id]
        if n_gt == 0:
            continue

        tp = torch.zeros(len(preds))
        fp = torch.zeros(len(preds))

        for i, (score, image_idx, box) in enumerate(preds):
            gt_entry = all_gts_by_class[class_id].get(image_idx)
            if gt_entry is None or len(gt_entry["boxes"]) == 0:
                fp[i] = 1
                continue

            pred_box = torch.tensor([box], dtype=torch.float32)
            gt_boxes_t = torch.tensor(gt_entry["boxes"], dtype=torch.float32)
            ious = box_iou(pred_box, gt_boxes_t)[0]
            best_iou, best_idx = ious.max(dim=0)

            if best_iou >= iou_thresh and not gt_entry["matched"][best_idx]:
                tp[i] = 1
                gt_entry["matched"][best_idx] = True
            else:
                fp[i] = 1

        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        recalls = tp_cum / n_gt
        precisions = tp_cum / (tp_cum + fp_cum).clamp(min=1e-9)

        aps[class_id] = compute_ap(recalls.tolist(), precisions.tolist())

    return aps


def main():
    parser = argparse.ArgumentParser(description="حساب mAP@0.5 لنموذج A-SSD مدرَّب")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", default="NORMAL", choices=["ECONOMY", "NORMAL", "TURBO"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = VOCDataset(args.data_root, split=args.split)

    ckpt = torch.load(args.checkpoint, map_location=device)
    # anchor_wh (إن وُجدت) يجب تحميلها مع الأوزان دائماً: رؤوس التوقع تعلّمت
    # إزاحات نسبية لأبعاد anchors محدَّدة وقت التدريب (K-Means أو الصيغة
    # الافتراضية) - استخدام أبعاد مختلفة هنا يُبطل معنى تلك الإزاحات.
    model = AdaptiveSSD(num_classes=len(VOC_CLASSES), anchor_wh=ckpt.get("anchor_wh")).to(device)
    model.load_state_dict(ckpt.get("model", ckpt))

    print(f"تقييم على {len(dataset)} صورة ({args.split}) بوضع {args.mode}...")
    aps = evaluate(model, dataset, mode=args.mode, device=device)

    print("\n--- الدقة لكل صنف (AP) ---")
    for class_id, ap in sorted(aps.items()):
        print(f"  {VOC_CLASSES[class_id]:15s}: {ap * 100:.1f}%")

    mean_ap = sum(aps.values()) / max(len(aps), 1)
    print(f"\nmAP@0.5 = {mean_ap * 100:.2f}%  (المرجع في التقرير: 71.4% على وضع NORMAL)")


if __name__ == "__main__":
    main()
