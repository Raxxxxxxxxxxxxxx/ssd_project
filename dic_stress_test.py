"""
اختبار DIC (Dynamic Inference Controller) تحت حِمل حقيقي - المرحلة 4.

كل قياسات المشروع حتى الآن كانت بوضع تشغيل ثابت (benchmark.py/evaluate.py
يمرران --mode واحد صراحة). DIC نفسه - المساهمة الأصلية للمشروع - لم يكن له
أي دليل تجريبي: هل يتكيّف فعلاً تحت حمل متغيّر؟ هذا السكربت يشغّل استدلالاً
حقيقياً من كاميرا حية، ويسجّل على نفس المحور الزمني: نسبة استخدام CPU، الوضع
الذي اختاره DIC، وFPS الفعلي - أثناء تحميل المعالج تدريجياً يدوياً بـstress-ng
من طرفية أخرى بالتوازي.

التشغيل (على Jetson Nano / Raspberry Pi 4 فعلي، كاميرا متصلة):

    الطرفية 1 (هذا السكربت):
        python dic_stress_test.py --duration 180 --log dic_stress_log.csv

    الطرفية 2 (بالتوازي - حمل تدريجي على مراحل، عدّل الأنوية/المدة حسب الجهاز):
        sleep 30  && stress-ng --cpu 1 --cpu-load 50 --timeout 60s
        sleep 90  && stress-ng --cpu 2 --cpu-load 95 --timeout 60s
        # اترك آخر 30 ثانية بلا حمل لمشاهدة العودة لـTURBO

بعد الانتهاء، لرسم الشكل المطلوب لقسم DIC بالفصل الخامس:
    python plot_dic_stress.py dic_stress_log.csv --out dic_stress.png
"""

import argparse
import csv
import os
import time

import cv2

SAMPLE_HEADER = ["elapsed_s", "timestamp", "cpu_usage", "ram_usage", "mode", "fps_instant", "fps_rolling"]


def main():
    parser = argparse.ArgumentParser(description="اختبار DIC تحت حمل حقيقي مع كاميرا")
    parser.add_argument("--backend", default="onnx", choices=["onnx", "pytorch"])
    parser.add_argument("--exports-dir", default="exports", help="لـ backend=onnx")
    parser.add_argument("--checkpoint", default=None, help="مطلوب لـ backend=pytorch")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--duration", type=int, default=180, help="مدة الاختبار بالثواني")
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--event-log", default="dic_switch_events.csv",
                         help="سجل تبديلات DIC الفعلية (من AdaptiveEngine مباشرة)")
    parser.add_argument("--log", default="dic_stress_log.csv", help="سجل العينات الدورية (CPU/mode/FPS)")
    args = parser.parse_args()

    if args.backend == "onnx":
        from onnx_inference import ONNXAdaptiveInference
        inf = ONNXAdaptiveInference(exports_dir=args.exports_dir, conf_thresh=0.5)
    else:
        from inference import AdaptiveInference
        inf = AdaptiveInference(checkpoint=args.checkpoint)

    from adaptive_engine import EVENT_LOG_HEADER
    with open(args.event_log, "w", newline="") as f:
        csv.writer(f).writerow(EVENT_LOG_HEADER)
    inf.engine.event_log_path = args.event_log

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"تعذّر فتح الكاميرا index={args.camera}")

    log_file = open(args.log, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(SAMPLE_HEADER)

    print(f"بدء اختبار DIC تحت حمل لمدة {args.duration}s.")
    print("حمّل المعالج الآن تدريجياً من طرفية أخرى بـstress-ng (راجع تعليمات أعلى الملف).\n")

    start = time.time()
    last_sample = -1.0
    frame_times = []
    rolling_window = 30

    try:
        while True:
            elapsed = time.time() - start
            if elapsed >= args.duration:
                break

            ok, frame = cap.read()
            if not ok:
                print("تحذير: فشل قراءة فريم من الكاميرا - إيقاف الاختبار.")
                break

            t0 = time.time()
            _detections, status = inf.run_inference(frame)
            frame_dt = time.time() - t0
            frame_times.append(frame_dt)
            if len(frame_times) > rolling_window:
                frame_times.pop(0)

            fps_instant = 1.0 / frame_dt if frame_dt > 0 else 0.0
            fps_rolling = len(frame_times) / sum(frame_times) if sum(frame_times) > 0 else 0.0

            if elapsed - last_sample >= args.sample_interval:
                writer.writerow([
                    f"{elapsed:.2f}", time.strftime("%Y-%m-%dT%H:%M:%S"),
                    f"{status['cpu_usage']:.1f}", f"{status['ram_usage']:.1f}",
                    status["mode"], f"{fps_instant:.1f}", f"{fps_rolling:.1f}",
                ])
                log_file.flush()
                last_sample = elapsed
                print(f"  t={elapsed:6.1f}s | CPU={status['cpu_usage']:5.1f}% | "
                      f"mode={status['mode']:8s} | FPS(rolling)={fps_rolling:5.1f}")
    finally:
        cap.release()
        log_file.close()
        inf.engine.stop()

    print(f"\nانتهى الاختبار.")
    print(f"  عينات CPU/mode/FPS: {os.path.abspath(args.log)}")
    print(f"  أحداث تبديل DIC:    {os.path.abspath(args.event_log)}")
    print(f"\nارسم الشكل المطلوب: python plot_dic_stress.py {args.log} --out dic_stress.png")


if __name__ == "__main__":
    main()
