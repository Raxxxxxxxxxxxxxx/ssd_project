"""
خادم بث فيديو حي (Video Streaming Server) لعرض نتائج A-SSD عبر المتصفح.

المشكلة التي يحلها هذا الملف: أجهزة الحافة المستهدفة في المشروع
(Raspberry Pi / Jetson Nano) تُنشر غالباً بدون شاشة متصلة (headless)،
فاستخدام cv2.imshow() في run_camera.py غير عملي على هذه الأجهزة. الحل
القياسي هو بث الفيديو كـ MJPEG عبر HTTP، بحيث يمكن لأي متصفح على الشبكة
نفسها فتح الرابط ومشاهدة الكشف اللحظي مباشرة (هاتف، حاسوب آخر...).

التشغيل:
    python stream_server.py --source 0 --host 0.0.0.0 --port 8000
    ثم افتح: http://<عنوان-الجهاز>:8000

يعمل الاستدلال في Thread خلفي مستقل يقرأ الكاميرا باستمرار ويحدّث آخر
فريم مُعالَج، بينما يخدم Flask طلبات المتصفحين من هذا الفريم المشترك -
هذا يفصل سرعة التقاط الكاميرا عن سرعة تسليم الشبكة، ويسمح بعدة متصفحين
متزامنين دون مضاعفة الاستدلال.
"""

import argparse
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template_string

from inference import AdaptiveInference
from run_camera import draw_detections, MODE_COLORS
from utils import VOC_CLASSES

app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<title>A-SSD - بث الكشف اللحظي</title>
<style>
  body { background:#111; color:#eee; font-family: sans-serif; text-align:center; margin:0; padding:20px; }
  h1 { font-size: 1.3rem; margin-bottom: 4px; }
  #stats { color:#9ad; margin-bottom: 12px; font-size: 0.95rem; }
  img { max-width: 100%; border: 2px solid #333; border-radius: 6px; }
</style>
</head>
<body>
  <h1>Adaptive Edge-SSD — بث الكشف اللحظي (Live Stream)</h1>
  <div id="stats">جاري التحميل...</div>
  <img src="{{ video_url }}" alt="live stream">
  <script>
    async function refreshStats() {
      try {
        const r = await fetch('/status');
        const s = await r.json();
        document.getElementById('stats').innerText =
          `الوضع: ${s.mode} | الدقة: ${s.resolution[0]}x${s.resolution[1]} | ` +
          `CPU: ${s.cpu_usage.toFixed(0)}% | RAM: ${s.ram_usage.toFixed(0)}% | FPS: ${s.fps.toFixed(1)}`;
      } catch (e) { /* الخادم قيد الإقلاع بعد */ }
    }
    setInterval(refreshStats, 1000);
    refreshStats();
  </script>
</body>
</html>
"""


class StreamWorker:
    """يلتقط الفريمات ويشغّل الاستدلال في خيط خلفي مستقل عن خدمة HTTP."""

    def __init__(self, source, checkpoint=None):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"تعذّر فتح مصدر الفيديو: {source}")

        self.system = AdaptiveInference(num_classes=len(VOC_CLASSES), checkpoint=checkpoint)

        self._lock = threading.Lock()
        self._latest_jpeg = None
        self._latest_status = {"mode": "NORMAL", "resolution": (300, 300),
                                "cpu_usage": 0.0, "ram_usage": 0.0, "fps": 0.0}

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        prev_time = time.time()
        fps = 0.0

        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            detections, status = self.system.run_inference(frame)

            now = time.time()
            fps = fps * 0.9 + (1.0 / max(now - prev_time, 1e-6)) * 0.1
            prev_time = now

            color = MODE_COLORS.get(status["mode"], (255, 255, 255))
            info_text = f"MODE: {status['mode']} | FPS: {fps:.1f}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            draw_detections(frame, detections, VOC_CLASSES, color)

            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue

            with self._lock:
                self._latest_jpeg = jpeg.tobytes()
                self._latest_status = {**status, "fps": fps}

    def get_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    def get_status(self):
        with self._lock:
            return dict(self._latest_status)

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self.cap.release()


worker = None  # يُهيَّأ في main() بعد قراءة معطيات سطر الأوامر


def mjpeg_generator():
    boundary = b"--frame"
    while True:
        jpeg = worker.get_jpeg()
        if jpeg is not None:
            yield (boundary + b"\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(0.03)  # ~30 FPS كحد أقصى لتسليم الشبكة


@app.route("/")
def index():
    return render_template_string(INDEX_HTML, video_url="/video_feed")


@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    return jsonify(worker.get_status())


def main():
    global worker
    parser = argparse.ArgumentParser(description="خادم بث فيديو حي لنموذج A-SSD")
    parser.add_argument("--source", default="0", help="رقم الكاميرا (0) أو مسار فيديو أو رابط RTSP")
    parser.add_argument("--checkpoint", default=None, help="مسار نسخة النموذج المدرَّبة (اختياري)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    worker = StreamWorker(source=source, checkpoint=args.checkpoint)

    print(f"البث متاح على: http://{args.host}:{args.port}  (اضغط Ctrl+C للإيقاف)")
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        worker.stop()


if __name__ == "__main__":
    main()
