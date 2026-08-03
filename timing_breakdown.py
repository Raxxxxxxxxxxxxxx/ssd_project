"""
تفكيك زمن الاستجابة الكامل end-to-end لكل مرحلة منفصلة - بديل عن رقم FPS
واحد إجمالي لا يفسّر أين يُصرَف الوقت فعلياً. يقيس: التقاط الفريم من
الكاميرا، المعالجة الأولية (BGR->RGB + resize + normalize)، الاستدلال
(session.run)، وما بعد المعالجة (فك الترميز + NMS). كل مرحلة بمعزل عن
الأخرى، بمتوسط على عدد فريمات حقيقي من كاميرا حية - بلا نافذة عرض (headless)
حتى يعمل عبر SSH بلا شاشة.

الاستخدام:
    python3 timing_breakdown.py --mode NORMAL --seconds 30 --camera 0
"""

import argparse
import time

import cv2
import numpy as np
import onnxruntime as ort

from utils import VOC_CLASSES, IMAGENET_MEAN, IMAGENET_STD
from tflite_inference import softmax_numpy, postprocess_numpy, MODE_RESOLUTIONS


def main():
    parser = argparse.ArgumentParser(description="تفكيك زمن end-to-end لكل مرحلة")
    parser.add_argument("--exports-dir", default="exports")
    parser.add_argument("--mode", default="NORMAL", choices=["ECONOMY", "NORMAL", "TURBO"])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--conf-thresh", type=float, default=0.5)
    args = parser.parse_args()

    mode_lower = args.mode.lower()
    session = ort.InferenceSession(f"{args.exports_dir}/assd_{mode_lower}.onnx",
                                    providers=["CPUExecutionProvider"])
    anchors = np.load(f"{args.exports_dir}/assd_{mode_lower}_anchors.npy")
    input_name = session.get_inputs()[0].name
    resolution = MODE_RESOLUTIONS[args.mode]

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"تعذّر فتح الكاميرا index={args.camera}")

    timings = {"capture": [], "preprocess": [], "inference": [], "postprocess": []}
    n_frames = 0
    measure_start = None

    print(f"بدء القياس (وضع {args.mode}، دقة {resolution}) - إحماء {args.warmup} فريم ثم {args.seconds} ثانية قياس فعلي...")

    while True:
        if measure_start is not None and (time.time() - measure_start) > args.seconds:
            break
        t0 = time.time()
        ok, frame = cap.read()
        t1 = time.time()
        if not ok:
            print("تحذير: فشل قراءة فريم - إيقاف.")
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, resolution).astype(np.float32) / 255.0
        normalized = (resized - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        input_data = normalized.transpose(2, 0, 1)[np.newaxis, ...]
        t2 = time.time()

        outputs = session.run(None, {input_name: input_data})
        t3 = time.time()

        outputs_by_shape = {o.shape[-1]: o[0] for o in outputs}
        scores = softmax_numpy(outputs_by_shape[len(VOC_CLASSES)], axis=-1)
        _dets = postprocess_numpy(outputs_by_shape[4], scores, anchors, (w, h),
                                   conf_thresh=args.conf_thresh, nms_thresh=0.45, top_k=200)
        t4 = time.time()

        n_frames += 1
        if n_frames > args.warmup:
            if measure_start is None:
                measure_start = time.time()
            timings["capture"].append(t1 - t0)
            timings["preprocess"].append(t2 - t1)
            timings["inference"].append(t3 - t2)
            timings["postprocess"].append(t4 - t3)

    cap.release()

    print(f"\nعدد الفريمات المقاسة (بعد الإحماء): {len(timings['capture'])}\n")
    print(f"{'المرحلة':<15}{'متوسط (ms)':<14}{'% من الإجمالي':<14}")
    print("-" * 43)
    total_avg = sum(sum(v) / max(len(v), 1) for v in timings.values())
    for stage, vals in timings.items():
        avg_ms = (sum(vals) / max(len(vals), 1)) * 1000
        pct = (avg_ms / (total_avg * 1000)) * 100 if total_avg > 0 else 0
        print(f"{stage:<15}{avg_ms:<14.2f}{pct:<14.1f}")
    print("-" * 43)
    print(f"{'الإجمالي':<15}{total_avg*1000:<14.2f}{'100.0':<14}")
    print(f"\nFPS end-to-end (الإجمالي الكامل): {1.0/total_avg:.1f}")
    print(f"FPS استدلال فقط (كما بمقاييس benchmark.py السابقة): {1.0/(sum(timings['inference'])/max(len(timings['inference']),1)):.1f}")


if __name__ == "__main__":
    main()
