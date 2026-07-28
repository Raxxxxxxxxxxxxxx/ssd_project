"""
قياس أداء حقيقي (FPS/زمن استجابة) لنموذج A-SSD على أي جهاز - لا يحتاج
كاميرا ولا فيديو، فقط مُدخلات عشوائية بنفس شكل الصورة الحقيقية. مُصمَّم
ليُشغَّل مباشرة على Raspberry Pi 4 / Jetson Nano للمقارنة الحقيقية بجدولي
4-1 و4-2 بالتقرير.

الاستخدام:
    python benchmark.py --checkpoint checkpoints_v2/last.pth --mode NORMAL
    python benchmark.py --checkpoint checkpoints_v2/last.pth --mode all --device cuda
    python benchmark.py --backend onnx --exports-dir exports --mode all

ملاحظة: backend=tflite غير مدعوم عمداً هنا - الملفات المُصدَّرة حالياً
معطوبة (راجع RESULTS.md، العطل #5) فقياس سرعتها مضلِّل ما دامت النتائج
خاطئة أصلاً. استخدم backend=onnx بدلاً منه.
"""

import argparse
import time

from utils import VOC_CLASSES


def benchmark_pytorch(args):
    import torch
    from assd_model import AdaptiveSSD, MODE_CONFIG

    device = torch.device(args.device)
    print(f"الجهاز: {device} | torch {torch.__version__}", end="")
    if device.type == "cuda":
        print(f" | GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f" | أنوية CPU: {torch.get_num_threads()}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = AdaptiveSSD(num_classes=len(VOC_CLASSES), anchor_wh=ckpt.get("anchor_wh")).to(device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    def run(mode, resolution, warmup, iters):
        x = torch.randn(1, 3, resolution[1], resolution[0], device=device)
        with torch.no_grad():
            for _ in range(warmup):
                model(x, mode=mode)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(iters):
                model(x, mode=mode)
            if device.type == "cuda":
                torch.cuda.synchronize()
        return (time.time() - t0) / iters

    return MODE_CONFIG, run


def benchmark_onnx(args):
    import numpy as np
    import onnxruntime as ort
    from assd_model import MODE_CONFIG

    print(f"الجهاز: CPU (ONNX Runtime) | onnxruntime {ort.__version__}")
    sessions = {}
    for mode in ("ECONOMY", "NORMAL", "TURBO"):
        path = f"{args.exports_dir}/assd_{mode.lower()}.onnx"
        sessions[mode] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def run(mode, resolution, warmup, iters):
        session = sessions[mode]
        input_name = session.get_inputs()[0].name
        x = np.random.randn(1, 3, resolution[1], resolution[0]).astype(np.float32)
        for _ in range(warmup):
            session.run(None, {input_name: x})
        t0 = time.time()
        for _ in range(iters):
            session.run(None, {input_name: x})
        return (time.time() - t0) / iters

    return MODE_CONFIG, run


def main():
    parser = argparse.ArgumentParser(description="قياس FPS/زمن استجابة A-SSD الحقيقي على هذا الجهاز")
    parser.add_argument("--backend", default="pytorch", choices=["pytorch", "onnx"])
    parser.add_argument("--checkpoint", default=None, help="مطلوب لـ backend=pytorch")
    parser.add_argument("--exports-dir", default="exports", help="لـ backend=onnx")
    parser.add_argument("--mode", default="all", choices=["ECONOMY", "NORMAL", "TURBO", "all"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    if args.backend == "pytorch":
        import torch
        if args.device == "cpu" and torch.cuda.is_available():
            args.device = "cuda"
        MODE_CONFIG, run = benchmark_pytorch(args)
    else:
        MODE_CONFIG, run = benchmark_onnx(args)

    modes = ["ECONOMY", "NORMAL", "TURBO"] if args.mode == "all" else [args.mode]

    print(f"\n{'الوضع':<10} {'الدقة':<12} {'زمن الاستجابة':<15} {'FPS':<8}")
    print("-" * 50)
    for mode in modes:
        resolution = MODE_CONFIG[mode]["resolution"]
        latency = run(mode, resolution, warmup=5, iters=args.iters)
        res = resolution
        print(f"{mode:<10} {f'{res[0]}x{res[1]}':<12} {latency*1000:>10.1f} ms   {1.0/latency:>6.1f}")

    print("\n(للمقارنة المرجعية بالتقرير: SSD-VGG16 الأصلي <3 FPS على RPi4، ~10 FPS على Jetson Nano)")


if __name__ == "__main__":
    main()
