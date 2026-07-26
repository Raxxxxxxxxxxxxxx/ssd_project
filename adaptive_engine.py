import time
import threading
import psutil

# دقة الإدخال (ARM) وعدد الطبقات المفعّلة (DFMS) لكل وضع - القسمان 3.4 و3.5
MODE_SETTINGS = {
    "TURBO":   {"resolution": (320, 320), "layers_count": 4},
    "NORMAL":  {"resolution": (300, 300), "layers_count": 3},
    "ECONOMY": {"resolution": (224, 224), "layers_count": 2},
}


class AdaptiveEngine:
    """
    Dynamic Inference Controller (DIC) - القسم 3.6 من التقرير.

    يراقب موارد الجهاز (CPU / RAM / الحرارة إن توفرت / البطارية) في **خيط
    مستقل** يعمل في الخلفية، ويحدّث الوضع كل `interval` ثانية دون أي حجب
    (blocking) لحلقة الاستدلال الرئيسية.

    ملاحظة مهمة: الإصدار السابق كان يستدعي psutil.cpu_percent(interval=0.1)
    مباشرة داخل حلقة معالجة كل فريم فيديو، ما يفرض توقفاً إجبارياً 100ms على
    كل إطار (سقف ~10 FPS بغض النظر عن سرعة النموذج فعلياً) - وهذا يناقض جوهر
    الهدف من المشروع (الأداء اللحظي على أجهزة الحافة). الحل: تشغيل المراقبة
    في Thread منفصل كما هو موصوف في الكود الوهمي بالتقرير (صفحة 26).
    """

    def __init__(self, interval=2.0, cpu_threshold_high=80, cpu_threshold_low=40):
        self.interval = interval
        self.cpu_threshold_high = cpu_threshold_high
        self.cpu_threshold_low = cpu_threshold_low

        self._lock = threading.Lock()
        self._status = {
            "mode": "NORMAL",
            "resolution": MODE_SETTINGS["NORMAL"]["resolution"],
            "layers_count": MODE_SETTINGS["NORMAL"]["layers_count"],
            "cpu_usage": 0.0,
            "ram_usage": 0.0,
            "battery": None,
            "temperature": None,
        }

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _read_battery(self):
        try:
            battery = psutil.sensors_battery()
            return battery.percent if battery is not None else None
        except Exception:
            return None  # غير متوفر على معظم أجهزة Raspberry Pi / Jetson Nano

    def _read_temperature(self):
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                return None
            # نأخذ أول قراءة حرارة متاحة (مثل "cpu_thermal" على Raspberry Pi)
            first_sensor = next(iter(temps.values()))
            return first_sensor[0].current if first_sensor else None
        except Exception:
            return None  # غير مدعوم على كل الأنظمة (مثل Windows/macOS أحياناً)

    def _decide(self, cpu, ram, battery, temperature):
        """قاعدة القرار الديناميكية (شكل 3-4 في التقرير)."""
        if battery is not None and battery <= 20:
            return "ECONOMY"
        if temperature is not None and temperature >= 80:
            return "ECONOMY"

        if cpu < self.cpu_threshold_low and ram < 60 and (temperature is None or temperature < 70):
            return "TURBO"
        if cpu < self.cpu_threshold_high and ram < 80 and (temperature is None or temperature < 80):
            return "NORMAL"
        return "ECONOMY"

    def _loop(self):
        while not self._stop_event.is_set():
            # interval هنا هو زمن أخذ العينة الفعلي لحساب متوسط استهلاك CPU
            # (0.5s كحد أقصى) وليس زمن حجب حلقة الفيديو، لأننا في خيط منفصل.
            cpu_usage = psutil.cpu_percent(interval=0.5)
            ram_usage = psutil.virtual_memory().percent
            battery = self._read_battery()
            temperature = self._read_temperature()

            mode = self._decide(cpu_usage, ram_usage, battery, temperature)
            settings = MODE_SETTINGS[mode]

            with self._lock:
                self._status = {
                    "mode": mode,
                    "resolution": settings["resolution"],
                    "layers_count": settings["layers_count"],
                    "cpu_usage": cpu_usage,
                    "ram_usage": ram_usage,
                    "battery": battery,
                    "temperature": temperature,
                }

            # ننام حتى نهاية الفترة المتبقية (interval الكلي بين قرارين، افتراضياً 2 ثانية)
            remaining = max(0.0, self.interval - 0.5)
            self._stop_event.wait(remaining)

    def get_current_mode(self):
        """
        استدعاء غير حاجب (non-blocking): يعيد فوراً آخر قرار اتخذه الخيط
        الخلفي، دون انتظار أي قياس جديد. آمن للاستدعاء من حلقة الفيديو
        على كل فريم بدون أي أثر على الأداء.
        """
        with self._lock:
            return dict(self._status)

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)


# اختبار المحرك بشكل مستقل
if __name__ == "__main__":
    engine = AdaptiveEngine(interval=2.0)
    try:
        for _ in range(5):
            status = engine.get_current_mode()
            print(f"Mode: {status['mode']:8s} | Res: {status['resolution']} | "
                  f"CPU: {status['cpu_usage']:.1f}% | RAM: {status['ram_usage']:.1f}% | "
                  f"Battery: {status['battery']} | Temp: {status['temperature']}")
            time.sleep(1.0)
    finally:
        engine.stop()
