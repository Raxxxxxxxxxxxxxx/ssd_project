import csv
import os
import time
import threading
import psutil

# دقة الإدخال (ARM) وعدد الطبقات المفعّلة (DFMS) لكل وضع - القسمان 3.4 و3.5
MODE_SETTINGS = {
    "TURBO":   {"resolution": (320, 320), "layers_count": 4},
    "NORMAL":  {"resolution": (300, 300), "layers_count": 3},
    "ECONOMY": {"resolution": (224, 224), "layers_count": 2},
}

EVENT_LOG_HEADER = [
    "timestamp", "old_mode", "new_mode", "dwell_seconds",
    "cpu_usage", "ram_usage", "battery", "temperature", "reason",
]


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

    Hysteresis + حد أدنى للبقاء (min dwell time): الإصدار السابق كان يستخدم
    نفس العتبة للدخول والخروج من كل وضع، فإذا استقر الحمل قرب عتبة (مثلاً
    CPU~40%) كان النظام قد يتذبذب بين وضعين كل دورة مراقبة. الآن: عتبة دخول
    أصعب من عتبة خروج لكل حد (hysteresis band)، بالإضافة لحد أدنى زمني
    (min_dwell_seconds) يمنع أي تبديل جديد قبل انقضائه - حتى لو كانت القراءة
    تبرر تبديلاً فورياً.
    """

    def __init__(self, interval=2.0,
                 cpu_turbo_enter=40, cpu_turbo_exit=55,
                 cpu_economy_enter=80, cpu_economy_exit=65,
                 min_dwell_seconds=5.0,
                 event_log_path=None):
        self.interval = interval
        self.cpu_turbo_enter = cpu_turbo_enter
        self.cpu_turbo_exit = cpu_turbo_exit
        self.cpu_economy_enter = cpu_economy_enter
        self.cpu_economy_exit = cpu_economy_exit
        self.min_dwell_seconds = min_dwell_seconds
        self.event_log_path = event_log_path

        self._lock = threading.Lock()
        self._current_mode = "NORMAL"
        self._mode_since = time.time()
        self._status = {
            "mode": "NORMAL",
            "resolution": MODE_SETTINGS["NORMAL"]["resolution"],
            "layers_count": MODE_SETTINGS["NORMAL"]["layers_count"],
            "cpu_usage": 0.0,
            "ram_usage": 0.0,
            "battery": None,
            "temperature": None,
            "reason": "initial",
            "time_in_mode": 0.0,
        }

        if self.event_log_path and not os.path.exists(self.event_log_path):
            with open(self.event_log_path, "w", newline="") as f:
                csv.writer(f).writerow(EVENT_LOG_HEADER)

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
        """قاعدة القرار الديناميكية (شكل 3-4 في التقرير) مع hysteresis: العتبة
        المستخدمة لكل حد تعتمد على الوضع الحالي (self._current_mode) - أصعب
        للدخول، أسهل نسبياً للبقاء. يعيد (الوضع المقترَح, السبب)."""
        if battery is not None and battery <= 20:
            return "ECONOMY", f"بطارية منخفضة ({battery:.0f}%<=20%)"
        if temperature is not None and temperature >= 80:
            return "ECONOMY", f"حرارة مرتفعة ({temperature:.0f}°C>=80°C)"

        was = self._current_mode
        ram_ok_turbo = ram < 60 and (temperature is None or temperature < 70)
        ram_ok_normal = ram < 80 and (temperature is None or temperature < 80)

        turbo_threshold = self.cpu_turbo_exit if was == "TURBO" else self.cpu_turbo_enter
        if cpu < turbo_threshold and ram_ok_turbo:
            band = "خروج" if was == "TURBO" else "دخول"
            return "TURBO", f"CPU={cpu:.0f}%<{turbo_threshold:.0f}% (عتبة {band})"

        economy_threshold = self.cpu_economy_exit if was == "ECONOMY" else self.cpu_economy_enter
        if cpu >= economy_threshold or not ram_ok_normal:
            band = "خروج" if was == "ECONOMY" else "دخول"
            reason = f"CPU={cpu:.0f}%>={economy_threshold:.0f}% (عتبة {band})" if cpu >= economy_threshold \
                else f"RAM={ram:.0f}% أو حرارة مرتفعة"
            return "ECONOMY", reason

        return "NORMAL", f"CPU={cpu:.0f}% ضمن نطاق NORMAL"

    def _log_switch(self, old_mode, new_mode, dwell_seconds, cpu, ram, battery, temperature, reason):
        if not self.event_log_path:
            return
        with open(self.event_log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                time.strftime("%Y-%m-%dT%H:%M:%S"), old_mode, new_mode, f"{dwell_seconds:.2f}",
                f"{cpu:.1f}", f"{ram:.1f}",
                "" if battery is None else f"{battery:.1f}",
                "" if temperature is None else f"{temperature:.1f}",
                reason,
            ])

    def _loop(self):
        while not self._stop_event.is_set():
            # interval هنا هو زمن أخذ العينة الفعلي لحساب متوسط استهلاك CPU
            # (0.5s كحد أقصى) وليس زمن حجب حلقة الفيديو، لأننا في خيط منفصل.
            cpu_usage = psutil.cpu_percent(interval=0.5)
            ram_usage = psutil.virtual_memory().percent
            battery = self._read_battery()
            temperature = self._read_temperature()
            now = time.time()

            proposed_mode, reason = self._decide(cpu_usage, ram_usage, battery, temperature)
            dwell_so_far = now - self._mode_since

            if proposed_mode != self._current_mode and dwell_so_far < self.min_dwell_seconds:
                # قرار بالتبديل لكن حد البقاء الأدنى لم ينقضِ بعد - نبقى بالوضع
                # الحالي هذه الدورة، ونعيد المحاولة بالدورة التالية.
                proposed_mode = self._current_mode
                reason = f"مؤجَّل (dwell={dwell_so_far:.1f}s<{self.min_dwell_seconds:.0f}s): {reason}"

            if proposed_mode != self._current_mode:
                self._log_switch(self._current_mode, proposed_mode, dwell_so_far,
                                  cpu_usage, ram_usage, battery, temperature, reason)
                self._current_mode = proposed_mode
                self._mode_since = now
                dwell_so_far = 0.0

            settings = MODE_SETTINGS[self._current_mode]
            with self._lock:
                self._status = {
                    "mode": self._current_mode,
                    "resolution": settings["resolution"],
                    "layers_count": settings["layers_count"],
                    "cpu_usage": cpu_usage,
                    "ram_usage": ram_usage,
                    "battery": battery,
                    "temperature": temperature,
                    "reason": reason,
                    "time_in_mode": dwell_so_far,
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


# اختبار المحرك بشكل مستقل - مرّر --log لتفعيل تسجيل CSV لكل تبديل فعلي
# (اختبار الحِمل الفعلي بـstress-ng على الجهاز يكون بسكربت dic_stress_test.py)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="اختبار مستقل لمحرك DIC")
    parser.add_argument("--log", default=None, help="مسار ملف CSV لتسجيل أحداث التبديل")
    parser.add_argument("--seconds", type=int, default=10)
    args = parser.parse_args()

    engine = AdaptiveEngine(interval=2.0, event_log_path=args.log)
    try:
        for _ in range(max(1, args.seconds)):
            status = engine.get_current_mode()
            print(f"Mode: {status['mode']:8s} | Res: {status['resolution']} | "
                  f"CPU: {status['cpu_usage']:.1f}% | RAM: {status['ram_usage']:.1f}% | "
                  f"Battery: {status['battery']} | Temp: {status['temperature']} | "
                  f"since {status['time_in_mode']:.1f}s | {status['reason']}")
            time.sleep(1.0)
    finally:
        engine.stop()
        if args.log:
            print(f"\nسجل أحداث التبديل: {args.log}")
