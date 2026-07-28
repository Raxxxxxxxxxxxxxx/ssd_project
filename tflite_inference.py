"""
مسار استدلال خفيف بدون PyTorch/torchvision إطلاقاً - يعتمد فقط على مُفسِّر
TFLite (ai_edge_litert) + numpy + OpenCV. مُصمَّم خصيصاً لأجهزة الحافة
محدودة الذاكرة (خاصة Raspberry Pi 4) حيث تثبيت PyTorch الكامل مكلف جداً على
الذاكرة والتخزين مقارنة بـ tflite-runtime الخفيف.

يتوقع بنية ملفات مطابقة لمخرجات export_onnx.py + export_tflite.py:
    exports/
        assd_<mode>_anchors.npy
        tflite_<mode>/assd_<mode>_full_integer_quant.tflite  (أو float16)

الواجهة (run_inference) مطابقة تماماً لـ AdaptiveInference في inference.py
(نفس الشكل: detections, status) حتى يعمل run_camera.py/stream_server.py مع
كلا المسارين (PyTorch أو TFLite) دون أي تعديل إضافي.
"""

import os

import cv2
import numpy as np

from adaptive_engine import AdaptiveEngine
from utils import VOC_CLASSES, IMAGENET_MEAN, IMAGENET_STD

MODE_RESOLUTIONS = {"ECONOMY": (224, 224), "NORMAL": (300, 300), "TURBO": (320, 320)}
VARIANCES = (0.1, 0.1, 0.2, 0.2)


def decode_numpy(loc_preds, anchors, variances=VARIANCES):
    """فك ترميز SSD القياسي بـ numpy بحت - يطابق utils.decode() رياضياً."""
    cxcy = anchors[:, :2] + loc_preds[:, :2] * variances[0] * anchors[:, 2:]
    wh = anchors[:, 2:] * np.exp(loc_preds[:, 2:] * variances[2])

    xmin = cxcy[:, 0] - wh[:, 0] / 2
    ymin = cxcy[:, 1] - wh[:, 1] / 2
    xmax = cxcy[:, 0] + wh[:, 0] / 2
    ymax = cxcy[:, 1] + wh[:, 1] / 2
    boxes = np.stack([xmin, ymin, xmax, ymax], axis=1)
    return np.clip(boxes, 0.0, 1.0)


def softmax_numpy(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def postprocess_numpy(loc_preds, cls_scores, anchors, orig_size,
                       conf_thresh=0.5, nms_thresh=0.45, top_k=200, pre_nms_top_k=400):
    """يطابق utils.postprocess() في الواجهة والمخرجات تماماً، لكن بـ numpy +
    cv2.dnn.NMSBoxes بدل torch + torchvision.ops.nms.

    pre_nms_top_k: حد أقصى للمرشَّحين قبل NMS لكل صنف. ضروري عملياً - قِسنا
    فعلياً أن نماذج INT8 (توزيع ثقة أقل حدّة من float) قد تُمرِّر آلاف
    المرشَّحين تحت عتبات ثقة منخفضة (كالمستخدمة أثناء حساب mAP)، وNMS بدون
    حد أقصى يتجمّد فعلياً (اختُبر: أكثر من 57 دقيقة بدون نتيجة على صورة
    واحدة). تقييد المرشَّحين لأفضل N بالثقة قبل NMS ممارسة قياسية في كل
    خطوط أنابيب الكشف (yolo/ssd الأصلية) وليس تجاوزاً مؤقتاً."""
    orig_w, orig_h = orig_size
    boxes_norm = decode_numpy(loc_preds, anchors)
    num_classes = cls_scores.shape[-1]
    detections = []

    for class_id in range(1, num_classes):  # نتجاوز الخلفية (0)
        scores = cls_scores[:, class_id]
        mask = scores > conf_thresh
        if not np.any(mask):
            continue

        boxes = boxes_norm[mask]
        class_scores = scores[mask]

        if class_scores.shape[0] > pre_nms_top_k:
            top_idx = np.argpartition(class_scores, -pre_nms_top_k)[-pre_nms_top_k:]
            boxes = boxes[top_idx]
            class_scores = class_scores[top_idx]

        # تحويل مُتَّجَه (vectorized) لصيغة (x, y, w, h) بالبكسل التي يتوقعها
        # cv2.dnn.NMSBoxes - حلقة Python لكل صندوق كانت بطيئة جداً عندما يمر
        # آلاف المرشَّحين من العتبة (كما يحدث فعلياً مع نماذج INT8 حيث الثقة
        # أقل حدّة، فيمر عدد أكبر بكثير من المرشَّحين) - قِسناه فعلياً: تجميد
        # كامل لأكثر من 57 دقيقة بدون نتيجة على تقييم 5823 صورة قبل هذا الإصلاح.
        x = (boxes[:, 0] * orig_w).astype(np.int32)
        y = (boxes[:, 1] * orig_h).astype(np.int32)
        w = np.maximum(1, ((boxes[:, 2] - boxes[:, 0]) * orig_w).astype(np.int32))
        h = np.maximum(1, ((boxes[:, 3] - boxes[:, 1]) * orig_h).astype(np.int32))
        pixel_boxes = np.stack([x, y, w, h], axis=1)

        keep = cv2.dnn.NMSBoxes(pixel_boxes.tolist(), class_scores.tolist(), conf_thresh, nms_thresh)
        for idx in np.array(keep).flatten():
            bx, by, bw, bh = pixel_boxes[idx]
            detections.append(((int(bx), int(by), int(bx + bw), int(by + bh)), class_id, float(class_scores[idx])))

    detections.sort(key=lambda d: d[2], reverse=True)
    return detections[:top_k]


class TFLiteAdaptiveInference:
    """بديل torch-free لـ inference.AdaptiveInference، بنفس الواجهة
    بالضبط (run_inference(frame) -> detections, status)."""

    def __init__(self, exports_dir="exports", quantized=True, conf_thresh=0.5, nms_thresh=0.45):
        from ai_edge_litert.interpreter import Interpreter

        self.engine = AdaptiveEngine()
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.interpreters = {}
        self.anchors = {}

        suffix = "full_integer_quant" if quantized else "float16"
        for mode in ("ECONOMY", "NORMAL", "TURBO"):
            mode_lower = mode.lower()
            model_path = os.path.join(exports_dir, f"tflite_{mode_lower}", f"assd_{mode_lower}_{suffix}.tflite")
            anchors_path = os.path.join(exports_dir, f"assd_{mode_lower}_anchors.npy")

            interp = Interpreter(model_path=model_path)
            interp.allocate_tensors()
            self.interpreters[mode] = interp
            self.anchors[mode] = np.load(anchors_path)

    def run_inference(self, frame):
        status = self.engine.get_current_mode()
        mode = status["mode"]
        resolution = MODE_RESOLUTIONS[mode]
        orig_h, orig_w = frame.shape[:2]

        interp = self.interpreters[mode]
        input_details = interp.get_input_details()[0]
        output_details = interp.get_output_details()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, resolution, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        normalized = (resized - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        input_data = normalized[np.newaxis, ...]  # NHWC

        if np.issubdtype(input_details["dtype"], np.integer):
            scale, zero_point = input_details["quantization"]
            input_data = np.round(input_data / scale + zero_point).astype(input_details["dtype"])

        interp.set_tensor(input_details["index"], input_data)
        interp.invoke()

        # نميّز مخرج loc عن cls حسب آخر بُعد (4 مقابل 21) بدل الاعتماد على
        # ترتيب ثابت - onnx2tf لا يضمن نفس الترتيب دائماً بين التصديرات.
        outputs = {}
        for d in output_details:
            val = interp.get_tensor(d["index"])[0]
            if np.issubdtype(d["dtype"], np.integer):
                scale, zero_point = d["quantization"]
                val = (val.astype(np.float32) - zero_point) * scale
            outputs[val.shape[-1]] = val

        loc = outputs[4]
        cls = outputs[len(VOC_CLASSES)]
        scores = softmax_numpy(cls, axis=-1)

        detections = postprocess_numpy(
            loc, scores, self.anchors[mode], (orig_w, orig_h),
            conf_thresh=self.conf_thresh, nms_thresh=self.nms_thresh,
        )
        return detections, status


if __name__ == "__main__":
    import numpy as _np

    inf = TFLiteAdaptiveInference()
    dummy_frame = _np.random.randint(0, 255, (480, 640, 3), dtype=_np.uint8)
    dets, status = inf.run_inference(dummy_frame)
    print(f"وضع: {status['mode']} | اكتشافات: {len(dets)} (نموذج مدرَّب فعلياً، لكن مُدخل عشوائي)")
