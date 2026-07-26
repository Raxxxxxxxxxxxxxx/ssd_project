import time
import cv2
from inference import AdaptiveInference
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


def main(source=0, checkpoint=None):
    print("Initializing Adaptive SSD Inference System...")
    class_names = VOC_CLASSES  # 20 صنف من Pascal VOC + الخلفية (21 صنفاً)
    system = AdaptiveInference(num_classes=len(class_names), checkpoint=checkpoint)

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
    main()
