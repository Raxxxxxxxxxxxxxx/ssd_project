"""
حساب mAP@0.5 حقيقي على مجموعة تحقق (val) بأسلوب Pascal VOC القياسي،
لمقارنة نتائج A-SSD المدرَّب فعلياً بالرقم المذكور في التقرير (%71.4).

الاستخدام:
    python evaluate.py --data-root /path/to/VOCdevkit/VOC2012 \
                        --checkpoint checkpoints/last.pth --mode NORMAL
"""

import argparse
from collections import defaultdict

import cv2
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


def evaluate(dataset, detect_fn, iou_thresh=0.5):
    """
    يحسب mAP@0.5 بغضّ النظر عن مصدر الاكتشافات - detect_fn(frame) -> قائمة
    (box, class_id, score) هي الواجهة الوحيدة المطلوبة. هذا يسمح بإعادة
    استخدام نفس منطق حساب AP لأي backend (PyTorch أو TFLite) دون تكرار.
    """
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

        detections = detect_fn(frame)

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
    parser.add_argument("--checkpoint", default=None, help="مطلوب لـ backend=pytorch")
    parser.add_argument("--mode", default="NORMAL", choices=["ECONOMY", "NORMAL", "TURBO"])
    parser.add_argument("--backend", default="pytorch", choices=["pytorch", "tflite", "onnx"])
    parser.add_argument("--exports-dir", default="exports", help="لـ backend=tflite/onnx")
    parser.add_argument("--tflite-float", action="store_true",
                         help="استخدام float16 بدل INT8 الكامل لـ backend=tflite")
    parser.add_argument("--conf-thresh", type=float, default=0.01)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = VOCDataset(args.data_root, split=args.split)

    if args.backend == "pytorch":
        ckpt = torch.load(args.checkpoint, map_location=device)
        # anchor_wh (إن وُجدت) يجب تحميلها مع الأوزان دائماً: رؤوس التوقع تعلّمت
        # إزاحات نسبية لأبعاد anchors محدَّدة وقت التدريب (K-Means أو الصيغة
        # الافتراضية) - استخدام أبعاد مختلفة هنا يُبطل معنى تلك الإزاحات.
        model = AdaptiveSSD(num_classes=len(VOC_CLASSES), anchor_wh=ckpt.get("anchor_wh")).to(device)
        model.load_state_dict(ckpt.get("model", ckpt))
        model.eval()

        from assd_model import MODE_CONFIG
        resolution = MODE_CONFIG[args.mode]["resolution"]

        @torch.no_grad()
        def detect_fn(frame):
            h, w = frame.shape[:2]
            input_tensor = preprocess(frame, resolution).to(device)
            loc_preds, cls_preds, anchors = model(input_tensor, mode=args.mode)
            return postprocess(loc_preds, cls_preds, anchors, orig_size=(w, h),
                                conf_thresh=args.conf_thresh, nms_thresh=0.45, top_k=200)
    elif args.backend == "onnx":
        from onnx_inference import ONNXAdaptiveInference
        from tflite_inference import softmax_numpy, postprocess_numpy, MODE_RESOLUTIONS

        onnx_inf = ONNXAdaptiveInference(exports_dir=args.exports_dir, conf_thresh=args.conf_thresh)
        onnx_inf.engine.stop()
        session = onnx_inf.sessions[args.mode]
        anchors = onnx_inf.anchors[args.mode]
        input_name = session.get_inputs()[0].name

        def detect_fn(frame):
            import numpy as np
            from utils import IMAGENET_MEAN, IMAGENET_STD

            resolution = MODE_RESOLUTIONS[args.mode]
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, resolution).astype(np.float32) / 255.0
            normalized = (resized - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
            input_data = normalized.transpose(2, 0, 1)[np.newaxis, ...]

            outputs = session.run(None, {input_name: input_data})
            outputs_by_shape = {o.shape[-1]: o[0] for o in outputs}
            scores = softmax_numpy(outputs_by_shape[len(VOC_CLASSES)], axis=-1)
            return postprocess_numpy(outputs_by_shape[4], scores, anchors, (w, h),
                                      conf_thresh=args.conf_thresh, nms_thresh=0.45, top_k=200)
    else:
        from tflite_inference import TFLiteAdaptiveInference

        tf_inf = TFLiteAdaptiveInference(exports_dir=args.exports_dir, quantized=not args.tflite_float,
                                          conf_thresh=args.conf_thresh)
        tf_inf.engine.stop()  # لا حاجة لخيط DIC هنا - الوضع ثابت طوال التقييم
        interp = tf_inf.interpreters[args.mode]
        anchors = tf_inf.anchors[args.mode]

        def detect_fn(frame):
            from tflite_inference import MODE_RESOLUTIONS, softmax_numpy, postprocess_numpy
            import cv2
            import numpy as np
            from utils import IMAGENET_MEAN, IMAGENET_STD

            resolution = MODE_RESOLUTIONS[args.mode]
            h, w = frame.shape[:2]
            input_details = interp.get_input_details()[0]
            output_details = interp.get_output_details()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, resolution).astype(np.float32) / 255.0
            normalized = (resized - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
            input_data = normalized[np.newaxis, ...]
            if np.issubdtype(input_details["dtype"], np.integer):
                scale, zp = input_details["quantization"]
                input_data = np.round(input_data / scale + zp).astype(input_details["dtype"])
            interp.set_tensor(input_details["index"], input_data)
            interp.invoke()

            outputs = {}
            for d in output_details:
                val = interp.get_tensor(d["index"])[0]
                if np.issubdtype(d["dtype"], np.integer):
                    scale, zp = d["quantization"]
                    val = (val.astype(np.float32) - zp) * scale
                outputs[val.shape[-1]] = val

            scores = softmax_numpy(outputs[len(VOC_CLASSES)], axis=-1)
            return postprocess_numpy(outputs[4], scores, anchors, (w, h),
                                      conf_thresh=args.conf_thresh, nms_thresh=0.45, top_k=200)

    print(f"تقييم على {len(dataset)} صورة ({args.split}) بوضع {args.mode}, backend={args.backend}...")
    aps = evaluate(dataset, detect_fn)

    print("\n--- الدقة لكل صنف (AP) ---")
    for class_id, ap in sorted(aps.items()):
        print(f"  {VOC_CLASSES[class_id]:15s}: {ap * 100:.1f}%")

    mean_ap = sum(aps.values()) / max(len(aps), 1)
    print(f"\nmAP@0.5 = {mean_ap * 100:.2f}%  (المرجع في التقرير: 71.4% على وضع NORMAL)")


if __name__ == "__main__":
    main()
