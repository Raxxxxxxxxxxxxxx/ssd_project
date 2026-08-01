"""
يرسم شكل "إثبات التكيّف" المطلوب لقسم DIC بالفصل الخامس: CPU% والوضع
المختار وFPS الفعلي على محور زمني مشترك، من سجل dic_stress_test.py.

ثلاث لوحات فرعية بمحور زمني واحد (بدل محور Y مزدوج على نفس اللوحة، لأن
CPU% وFPS بمقياسين مختلفين تماماً) - أوضح للقراءة وأصدق تمثيلاً.

الاستخدام:
    python plot_dic_stress.py dic_stress_log.csv --out dic_stress.png
"""

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODE_ORDER = ["ECONOMY", "NORMAL", "TURBO"]
MODE_Y = {name: i for i, name in enumerate(MODE_ORDER)}
MODE_COLOR = {"ECONOMY": "#8a94a6", "NORMAL": "#d98a2b", "TURBO": "#1f8a70"}

CPU_COLOR = "#3b6fa0"
FPS_COLOR = "#1f8a70"


def load_log(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "t": float(row["elapsed_s"]),
                "cpu": float(row["cpu_usage"]),
                "mode": row["mode"],
                "fps": float(row["fps_rolling"]),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="رسم CPU%/mode/FPS من سجل dic_stress_test.py")
    parser.add_argument("csv_path")
    parser.add_argument("--out", default="dic_stress.png")
    args = parser.parse_args()

    rows = load_log(args.csv_path)
    if not rows:
        raise SystemExit(f"لا توجد بيانات في {args.csv_path}")

    t = [r["t"] for r in rows]
    cpu = [r["cpu"] for r in rows]
    fps = [r["fps"] for r in rows]
    mode_y = [MODE_Y[r["mode"]] for r in rows]

    fig, (ax_cpu, ax_mode, ax_fps) = plt.subplots(
        3, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [1.1, 0.8, 1.1]},
    )
    fig.suptitle("استجابة DIC لحمل CPU متغيّر (اختبار حقيقي على الجهاز)", fontsize=13)

    ax_cpu.plot(t, cpu, color=CPU_COLOR, linewidth=1.8)
    ax_cpu.fill_between(t, cpu, color=CPU_COLOR, alpha=0.12)
    ax_cpu.set_ylabel("CPU %")
    ax_cpu.set_ylim(0, 100)
    ax_cpu.grid(axis="y", alpha=0.25)

    # لوحة الوضع: تظليل خلفي بلون كل وضع لكل مقطع زمني + خط درجي فوقه
    seg_start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i]["mode"] != rows[seg_start]["mode"]:
            mode = rows[seg_start]["mode"]
            t_end = t[i] if i < len(rows) else t[-1] + (t[-1] - t[-2] if len(t) > 1 else 1)
            ax_mode.axvspan(t[seg_start], t_end, color=MODE_COLOR[mode], alpha=0.18, linewidth=0)
            seg_start = i
    ax_mode.step(t, mode_y, where="post", color="#2b2f36", linewidth=1.8)
    ax_mode.set_yticks(list(MODE_Y.values()))
    ax_mode.set_yticklabels(MODE_ORDER)
    ax_mode.set_ylabel("الوضع")
    ax_mode.set_ylim(-0.5, 2.5)
    ax_mode.grid(axis="y", alpha=0.25)

    ax_fps.plot(t, fps, color=FPS_COLOR, linewidth=1.8)
    ax_fps.fill_between(t, fps, color=FPS_COLOR, alpha=0.12)
    ax_fps.set_ylabel("FPS (متوسط متحرك)")
    ax_fps.set_xlabel("الزمن منذ بدء الاختبار (ثانية)")
    ax_fps.set_ylim(bottom=0)
    ax_fps.grid(axis="y", alpha=0.25)

    n_switches = sum(1 for i in range(1, len(rows)) if rows[i]["mode"] != rows[i - 1]["mode"])
    fig.text(0.5, 0.01, f"عدد تبديلات الوضع المرصودة بالعيّنات: {n_switches}"
                         f"  (سجل الأحداث الدقيق بالثواني في dic_switch_events.csv)",
              ha="center", fontsize=9, color="#5c6b79")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(args.out, dpi=160)
    print(f"حُفظ الشكل في: {args.out}")


if __name__ == "__main__":
    main()
