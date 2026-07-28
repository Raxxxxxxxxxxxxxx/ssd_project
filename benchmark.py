"""
قياس أداء حقيقي (FPS/زمن استجابة) لنموذج A-SSD على أي جهاز - لا يحتاج
كاميرا ولا فيديو، فقط مُدخلات عشوائية بنفس شكل الصورة الحقيقية. مُصمَّم
ليُشغَّل مباشرة على Raspberry Pi 4 / Jetson Nano للمقارنة الحقيقية بجدولي
4-1 و4-2 بالتقرير.

الاستخدام:
    python benchmark.py --checkpoint checkpoints_v2/last.pth --mode NORMAL
    python benchmark.py --checkpoint checkpoints_v2/last.pth --mode all --device cuda
"""

import argparse
import time

import torch

from assd_model import AdaptiveSSD, MODE_CONFIG
from utils import VOC_CLASSES


def benchmark_mode(model, mode, device, warmup=5, iters=30):
    resolution = MODE_CONFIG[mode]["resolution"]
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
        dt = (time.time() - t0) / iters

    return dt, 1.0 / dt


def main():
    parser = argparse.ArgumentParser(description="قياس FPS/زمن استجابة A-SSD الحقيقي على هذا الجهاز")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", default="all", choices=["ECONOMY", "NORMAL", "TURBO", "all"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

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

    modes = ["ECONOMY", "NORMAL", "TURBO"] if args.mode == "all" else [args.mode]

    print(f"\n{'الوضع':<10} {'الدقة':<12} {'زمن الاستجابة':<15} {'FPS':<8}")
    print("-" * 50)
    for mode in modes:
        latency, fps = benchmark_mode(model, mode, device, iters=args.iters)
        res = MODE_CONFIG[mode]["resolution"]
        print(f"{mode:<10} {f'{res[0]}x{res[1]}':<12} {latency*1000:>10.1f} ms   {fps:>6.1f}")

    print("\n(للمقارنة المرجعية بالتقرير: SSD-VGG16 الأصلي <3 FPS على RPi4، ~10 FPS على Jetson Nano)")


if __name__ == "__main__":
    main()
