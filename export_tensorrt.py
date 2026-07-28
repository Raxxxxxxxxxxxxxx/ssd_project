"""
تصدير A-SSD إلى TensorRT (Jetson Nano - القسم 4.4 من التقرير).

ملاحظة صادقة مهمة: هذا السكربت **لم يُختبَر فعلياً على هذا الجهاز** لأن
TensorRT يحتاج NVIDIA GPU (أو Jetson Nano نفسه) مع مكتبات NVIDIA مثبَّتة -
غير متوفرة على جهاز التطوير هذا (CPU فقط). الكود مكتوب بعناية ومطابق
لتوثيق NVIDIA الرسمي، لكن يجب اختباره فعلياً على Jetson Nano قبل الوثوق
بنتائجه - لا تفترض نجاحه لمجرد أنه لم يُصادف خطأ لغوياً.

المتطلبات على Jetson Nano (عادة مثبَّتة مسبقاً مع JetPack):
    - trtexec (يأتي مع تثبيت TensorRT)
    - أو حزمة بايثون tensorrt + pycuda لتشغيل الاستدلال مباشرة من بايثون

خطوتان:
  1) تصدير ONNX (على أي جهاز، بما فيه هذا الجهاز): export_onnx.py
  2) بناء محرك TensorRT (يجب أن يتم على Jetson Nano نفسه، لأن محرك TensorRT
     مبنيّ خصيصاً لعتاد GPU الذي وُلِّد عليه - لا يُنقَل بين أجهزة مختلفة)

الاستخدام (على Jetson Nano، بعد نقل ملف .onnx إليه):
    python export_tensorrt.py --onnx assd_normal.onnx --output assd_normal.engine
"""

import argparse
import shutil
import subprocess
import sys


def build_with_trtexec(onnx_path, output_path, fp16=True, workspace_mb=1024):
    """يبني محرك TensorRT عبر أداة trtexec سطر الأوامر (الطريقة الأبسط،
    تحتاج TensorRT مثبَّتاً على PATH - متوفر افتراضياً على Jetson Nano
    مع JetPack)."""
    if shutil.which("trtexec") is None:
        raise SystemExit(
            "trtexec غير موجود على PATH. هذا متوقّع إن كنت تشغّل هذا خارج "
            "Jetson Nano/جهاز فيه TensorRT مثبَّت. راجع تعليقات الملف أعلاه."
        )

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        f"--workspace={workspace_mb}",
    ]
    if fp16:
        cmd.append("--fp16")

    print("تشغيل:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-3000:])
    if result.returncode != 0:
        print(result.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"فشل بناء محرك TensorRT (exit code {result.returncode})")

    print(f"تم بناء المحرك بنجاح: {output_path}")


def verify_engine(engine_path):
    """التحقق الفعلي (يعمل فقط على جهاز فيه GPU + TensorRT + pycuda) -
    يُشغِّل استدلالاً حقيقياً على مدخل عشوائي ويتأكد أن المخرجات منطقية،
    تماماً كما فعلنا مع ONNX وTFLite - لا نكتفي بافتراض أن البناء الناجح
    يعني تشغيلاً صحيحاً."""
    try:
        import numpy as np
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        import tensorrt as trt
    except ImportError:
        print(
            "لا يمكن التحقق هنا: مكتبات tensorrt/pycuda غير مثبَّتة على هذا "
            "الجهاز (متوقّع - هي متوفرة فقط على Jetson Nano/أجهزة NVIDIA). "
            "شغّل هذه الدالة على Jetson Nano بعد بناء المحرك للتأكد الفعلي."
        )
        return

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    input_shape = engine.get_tensor_shape(engine.get_tensor_name(0))
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    d_input = cuda.mem_alloc(dummy_input.nbytes)
    cuda.memcpy_htod(d_input, dummy_input)
    context.execute_v2([int(d_input)])

    print("التحقق ناجح: محرك TensorRT يعمل فعلياً (استدلال حقيقي بدون عطل).")


def main():
    parser = argparse.ArgumentParser(description="بناء محرك TensorRT من ONNX (لـJetson Nano)")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-fp16", action="store_true", help="تعطيل FP16 (استخدام FP32 الكامل)")
    parser.add_argument("--workspace-mb", type=int, default=1024)
    args = parser.parse_args()

    output_path = args.output or args.onnx.replace(".onnx", ".engine")

    build_with_trtexec(args.onnx, output_path, fp16=not args.no_fp16, workspace_mb=args.workspace_mb)
    verify_engine(output_path)


if __name__ == "__main__":
    main()
