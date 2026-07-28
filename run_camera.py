import argparse
import time

import cv2

from utils import VOC_CLASSES

MODE_COLORS = {
    "TURBO": (0, 255, 0),
    "NORMAL": (255, 165, 0),
    "ECONOMY": (0, 0, 255),
}


def draw_detections(frame, detections, class_names, color):
    for (xmin, ymin, xmax, ymax), class_id, score in detections:
        label = f"{class_names[class_id]}: {score * 100:.0f}%"
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, label, (xmin, max(0, ymin - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def build_system(backend, checkpoint, exports_dir, tflite_float):
    """يبني نظام الاستدلال حسب الـbackend المختار - كلاهما يشارك نفس واجهة
    run_inference(frame) -> (detections, status) فيعمل باقي الكود بلا تغيير."""
    class_names = VOC_CLASSES
    if backend == "pytorch":
        from inference import AdaptiveInference
        return AdaptiveInference(num_classes=len(class_names), checkpoint=checkpoint)
    elif backend == "onnx":
        from onnx_inference import ONNXAdaptiveInference
        return ONNXAdaptiveInference(exports_dir=exports_dir)
    else:
        # ملاحظة: مسار TFLite (onnx2tf) فيه خطأ مؤكَّد يُبعثر ترتيب المخرجات
        # وينتج mAP@0.5 ≈ 0% رغم عدم وجود عطل تنفيذي - راجع RESULTS.md.
        # استخدم backend="onnx" بدلاً منه إلى حين إصلاحه.
        from tflite_inference import TFLiteAdaptiveInference
        return TFLiteAdaptiveInference(exports_dir=exports_dir, quantized=not tflite_float)


def main(source=0, checkpoint=None, backend="pytorch", exports_dir="exports", tflite_float=False):
    print(f"Initializing Adaptive SSD Inference System (backend={backend})...")
    class_names = VOC_CLASSES  # 20 صنف من Pascal VOC + الخلفية (21 صنفاً)
    system = build_system(backend, checkpoint, exports_dir, tflite_float)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Camera started. Press 'q' to exit.")

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # 1. تشغيل الاستدلال التكيفي (يستخدم آخر وضع قرره DIC في الخلفية)
        detections, status = system.run_inference(frame)

        # 2. حساب معدل الإطارات الفعلي للحلقة الكاملة (كاميرا + استدلال + رسم)
        now = time.time()
        instant_fps = 1.0 / max(now - prev_time, 1e-6)
        fps = fps * 0.9 + instant_fps * 0.1  # متوسط متحرك لتفادي الاهتزاز
        prev_time = now

        # 3. معلومات النظام والوضع الحالي
        mode = status["mode"]
        color = MODE_COLORS.get(mode, (255, 255, 255))
        info_text = (f"MODE: {mode} | Res: {status['resolution'][0]}x{status['resolution'][1]} | "
                     f"CPU: {status['cpu_usage']:.0f}% | RAM: {status['ram_usage']:.0f}% | "
                     f"FPS: {fps:.1f}")
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 4. رسم الاكتشافات الفعلية الناتجة عن النموذج (وليست صندوقاً وهمياً)
        draw_detections(frame, detections, class_names, color)

        # 5. عرض الفريم
        cv2.imshow("A-SSD Adaptive Real-time Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="عرض كشف A-SSD الحي محلياً (نافذة OpenCV)")
    parser.add_argument("--source", default="0", help="رقم الكاميرا (0) أو مسار فيديو")
    parser.add_argument("--backend", default="pytorch", choices=["pytorch", "onnx", "tflite"])
    parser.add_argument("--checkpoint", default=None, help="لـ backend=pytorch")
    parser.add_argument("--exports-dir", default="exports", help="لـ backend=tflite")
    parser.add_argument("--tflite-float", action="store_true", help="float16 بدل INT8 لـ backend=tflite")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    main(source=source, checkpoint=args.checkpoint, backend=args.backend,
         exports_dir=args.exports_dir, tflite_float=args.tflite_float)
