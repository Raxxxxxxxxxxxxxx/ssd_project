"""
مسار استدلال خفيف بدون PyTorch/torchvision - يعتمد على onnxruntime + numpy +
OpenCV فقط. بديل عن tflite_inference.py بعد اكتشاف عطل حقيقي في تحويل
onnx2tf (ONNX->TFLite) يُبعثر ترتيب المخرجات وينتج mAP@0.5 ≈ 0% رغم عدم وجود
أي عطل تنفيذي (تحقّقنا: نفس المدخل يُنتج قيماً مختلفة تماماً بين ONNX
وTFLite float16، رغم أن الفارق يجب أن يكون شبه معدوم لو كان التحويل صحيحاً).

onnxruntime نفسه **متحقَّق منه فعلياً** (فرق أقصى 5e-6 عن PyTorch - راجع
export_onnx.py)، ومتوفر كعجلات (wheels) لـARM64 (Raspberry Pi 4/Jetson Nano
بنظام 64-bit، وهو الشائع حالياً) دون الحاجة لبيئة Python منفصلة أو TensorFlow
الثقيلة - فقط `pip install onnxruntime numpy opencv-python`.

يتوقع بنية ملفات مطابقة لمخرجات export_onnx.py:
    exports/assd_<mode>.onnx
    exports/assd_<mode>_anchors.npy
"""

import os

import cv2
import numpy as np
import onnxruntime as ort

from adaptive_engine import AdaptiveEngine
from tflite_inference import decode_numpy, softmax_numpy, postprocess_numpy, MODE_RESOLUTIONS
from utils import VOC_CLASSES, IMAGENET_MEAN, IMAGENET_STD


class ONNXAdaptiveInference:
    """بديل torch-free لـ inference.AdaptiveInference عبر onnxruntime، بنفس
    الواجهة بالضبط (run_inference(frame) -> detections, status)."""

    def __init__(self, exports_dir="exports", conf_thresh=0.5, nms_thresh=0.45):
        self.engine = AdaptiveEngine()
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.sessions = {}
        self.anchors = {}

        for mode in ("ECONOMY", "NORMAL", "TURBO"):
            mode_lower = mode.lower()
            model_path = os.path.join(exports_dir, f"assd_{mode_lower}.onnx")
            anchors_path = os.path.join(exports_dir, f"assd_{mode_lower}_anchors.npy")

            self.sessions[mode] = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.anchors[mode] = np.load(anchors_path)

    def run_inference(self, frame):
        status = self.engine.get_current_mode()
        mode = status["mode"]
        resolution = MODE_RESOLUTIONS[mode]
        orig_h, orig_w = frame.shape[:2]

        session = self.sessions[mode]
        input_name = session.get_inputs()[0].name

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, resolution, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        normalized = (resized - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        input_data = normalized.transpose(2, 0, 1)[np.newaxis, ...]  # NHWC -> NCHW (ONNX يتوقع NCHW)

        outputs = session.run(None, {input_name: input_data})
        outputs_by_shape = {o.shape[-1]: o[0] for o in outputs}

        loc = outputs_by_shape[4]
        cls = outputs_by_shape[len(VOC_CLASSES)]
        scores = softmax_numpy(cls, axis=-1)

        detections = postprocess_numpy(
            loc, scores, self.anchors[mode], (orig_w, orig_h),
            conf_thresh=self.conf_thresh, nms_thresh=self.nms_thresh,
        )
        return detections, status


if __name__ == "__main__":
    inf = ONNXAdaptiveInference(conf_thresh=0.01)
    inf.engine.stop()

    import cv2 as _cv2
    frame = _cv2.imread("data/VOCdevkit/VOC2012/JPEGImages/2008_000008.jpg")
    dets, status = inf.run_inference(frame)
    print(f"وضع: {status['mode']} | اكتشافات: {len(dets)}")
    for box, cls_id, score in dets[:5]:
        print(f"  {VOC_CLASSES[cls_id]}: {score:.2f} box={box}")
