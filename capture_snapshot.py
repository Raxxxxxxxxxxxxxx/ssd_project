"""
يلتقط فريماً واحداً حياً من الكاميرا، يشغّل استدلالاً حقيقياً (checkpoint v3)،
يرسم الاكتشافات، ويحفظ الصورة - بلا نافذة عرض (headless)، للاستخدام عبر SSH.

الاستخدام:
    python3 capture_snapshot.py --mode NORMAL --out evidence/live_01.jpg
"""

import argparse

import cv2
import numpy as np
import onnxruntime as ort

from utils import VOC_CLASSES, IMAGENET_MEAN, IMAGENET_STD
from tflite_inference import softmax_numpy, postprocess_numpy, MODE_RESOLUTIONS

MODE_COLOR_BGR = {"TURBO": (0, 200, 0), "NORMAL": (0, 165, 255), "ECONOMY": (60, 60, 220)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports-dir", default="exports")
    parser.add_argument("--mode", default="NORMAL", choices=["ECONOMY", "NORMAL", "TURBO"])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf-thresh", type=float, default=0.35)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup", type=int, default=8, help="فريمات تجاهَل قبل اللقطة (تثبيت التعريض/التركيز)")
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                         help="تصحيح دوران الكاميرا إن كانت مركَّبة بزاوية (180 شائع)")
    args = parser.parse_args()

    mode_lower = args.mode.lower()
    session = ort.InferenceSession(f"{args.exports_dir}/assd_{mode_lower}.onnx", providers=["CPUExecutionProvider"])
    anchors = np.load(f"{args.exports_dir}/assd_{mode_lower}_anchors.npy")
    input_name = session.get_inputs()[0].name
    resolution = MODE_RESOLUTIONS[args.mode]

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"تعذّر فتح الكاميرا index={args.camera}")

    frame = None
    for _ in range(args.warmup + 1):
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("فشل التقاط فريم من الكاميرا")
    cap.release()

    if args.rotate == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif args.rotate == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif args.rotate == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, resolution).astype(np.float32) / 255.0
    normalized = (resized - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    input_data = normalized.transpose(2, 0, 1)[np.newaxis, ...]

    outputs = session.run(None, {input_name: input_data})
    outputs_by_shape = {o.shape[-1]: o[0] for o in outputs}
    scores = softmax_numpy(outputs_by_shape[len(VOC_CLASSES)], axis=-1)
    dets = postprocess_numpy(outputs_by_shape[4], scores, anchors, (w, h),
                              conf_thresh=args.conf_thresh, nms_thresh=0.45, top_k=200)

    color = MODE_COLOR_BGR[args.mode]
    for (xmin, ymin, xmax, ymax), class_id, score in dets:
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        label = f"{VOC_CLASSES[class_id]} {score*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (xmin, max(0, ymin - th - 10)), (xmin + tw + 6, ymin), color, -1)
        cv2.putText(frame, label, (xmin + 3, max(14, ymin - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"[{args.mode}] Jetson Nano - {len(dets)} detections",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    cv2.imwrite(args.out, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"حُفظت: {args.out}  ({len(dets)} اكتشاف)")
    for (box, class_id, score) in dets:
        print(f"  - {VOC_CLASSES[class_id]}: {score*100:.0f}%")


if __name__ == "__main__":
    main()
