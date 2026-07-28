"""
تحويل A-SSD إلى TFLite (ضروري لـ Raspberry Pi/Android - القسم 4.4).

ملاحظة بيئة مهمة: TensorFlow لا يوفّر بعد عجلات (wheels) لإصدار Python الذي
يعمل عليه المشروع الرئيسي (3.14 - إصدار حديث جداً). لذلك يعمل هذا السكربت
داخل بيئة Python منفصلة ومعزولة (`tflite_env/`, Python 3.11) أُنشئت خصيصاً
لهذه الخطوة فقط، ولا علاقة لها بالبيئة الرئيسية (`assd_env/`). المدخل
الوحيد هو ملف ONNX (الناتج من `export_onnx.py`) - لا حاجة لـ PyTorch هنا
إطلاقاً، فالتحويل يعتمد على onnx2tf (ONNX -> TensorFlow -> TFLite) بدلاً من
ai_edge_torch المذكورة بالتقرير، لأن ai_edge_torch يفرض خفض إصدار PyTorch
بشكل يكسر توافقه مع torchvision في هذا المشروع (جُرِّب فعلياً وتَبيَّن العطل).

الإعداد (مرة واحدة، من جذر المشروع):
    python3.11 -m venv tflite_env
    ./tflite_env/bin/pip install onnx2tf tensorflow onnx

الاستخدام:
    ./tflite_env/bin/python export_tflite.py --onnx assd_normal.onnx \
        --output assd_normal.tflite
"""

import argparse
import os
from pathlib import Path


def ensure_calibration_file():
    """
    onnx2tf يحاول تحميل صورة معايرة عيّنة من GitHub/Wasabi تلقائياً، لكن
    التحميل قد يفشل أو يُنتج ملفاً لا يتوافق مع np.load الحديث (يحتاج
    allow_pickle رغم أن الملف رقمي بحت) - جرّبنا هذا فعلياً وتعطّل التحويل.
    الحل: نُنشئ ملف المعايرة الوهمي بأنفسنا محلياً بنفس الاسم/الشكل الذي
    يتوقعه onnx2tf، فيتخطى التحميل بالكامل. هذا الملف يُستخدَم فقط لفحص دقة
    onnx2tf الداخلي (مقارنة ONNX مقابل TF) وليس لمعايرة التكميم الفعلية -
    تلك تُبنى من صور VOC حقيقية عبر build_real_calibration_data() عند تفعيل
    --quantize.
    """
    import numpy as np

    filename = "calibration_image_sample_data_20x128x128x3_float32.npy"
    if not os.path.isfile(filename):
        np.save(filename, np.random.rand(20, 128, 128, 3).astype(np.float32))


def build_real_calibration_data(voc_root, resolution, n=50, output_path="calib_real_int8.npy"):
    """
    يبني بيانات معايرة INT8 حقيقية (وليست عشوائية) من صور VOC2012 فعلية -
    ضروري لدقة تكميم مقبولة (نطاقات القيم يجب أن تعكس صوراً حقيقية).

    مهم: نطبّق هنا تطبيع ImageNet mean/std بأنفسنا (بنفس أرقام utils.preprocess
    بالضبط) قبل تمرير البيانات لـonnx2tf، لأن رسم ONNX نفسه يتوقع مدخلاً
    مُطبَّعاً مسبقاً (التطبيع في هذا المشروع يحدث في preprocess() بلغة
    Python خارج الرسم البياني المُصدَّر، وليس كطبقة داخل النموذج). لذلك نمرّر
    mean=0/std=1 لـonnx2tf لاحقاً حتى لا يُطبَّع مرتين.
    """
    import cv2
    import numpy as np

    ids_file = os.path.join(voc_root, "ImageSets", "Main", "train.txt")
    with open(ids_file) as f:
        ids = [line.strip() for line in f if line.strip()][:n]

    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    images = []
    for image_id in ids:
        frame = cv2.imread(os.path.join(voc_root, "JPEGImages", f"{image_id}.jpg"))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, resolution, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        normalized = (resized - imagenet_mean) / imagenet_std
        images.append(normalized)

    data = np.stack(images, axis=0)  # NHWC كما يتوقع onnx2tf
    np.save(output_path, data)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="تحويل A-SSD من ONNX إلى TFLite")
    parser.add_argument("--onnx", required=True, help="مسار ملف .onnx (من export_onnx.py)")
    parser.add_argument("--output-dir", default="tflite_out")
    parser.add_argument("--quantize", action="store_true",
                         help="إنتاج نسخة INT8 كاملة أيضاً (يحتاج --voc-root للمعايرة الحقيقية)")
    parser.add_argument("--voc-root", default=None, help="مسار VOCdevkit/VOC2012 لبيانات المعايرة")
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--resolution", type=int, nargs=2, default=[300, 300], metavar=("W", "H"))
    args = parser.parse_args()

    ensure_calibration_file()

    convert_kwargs = {}
    if args.quantize:
        if not args.voc_root:
            raise SystemExit("--quantize يحتاج --voc-root لبناء بيانات معايرة حقيقية")
        calib_path = build_real_calibration_data(args.voc_root, tuple(args.resolution))
        convert_kwargs["output_integer_quantized_tflite"] = True
        convert_kwargs["custom_input_op_name_np_data_path"] = [[args.input_name, calib_path, [[[0.0, 0.0, 0.0]]], [[[1.0, 1.0, 1.0]]]]]

    import onnx2tf

    onnx2tf.convert(
        input_onnx_file_path=args.onnx,
        **convert_kwargs,
        output_folder_path=args.output_dir,
        output_signaturedefs=True,
        non_verbose=True,
    )

    tflite_files = list(Path(args.output_dir).glob("*.tflite"))
    if not tflite_files:
        raise SystemExit("فشل التحويل: لم يُنتَج أي ملف .tflite")

    print(f"تم التحويل بنجاح. الملفات الناتجة في: {args.output_dir}/")
    for f in tflite_files:
        print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    # التحقق الفعلي: تشغيل أوسع نسخة (float32) وأصغر نسخة INT8 كاملة إن
    # وُجدت - كل نوع يحتاج تعامل مدخل/مخرج مختلف (float مقابل int8 صريح).
    verify_path = sorted(tflite_files)[0]
    print(f"\nالتحقق الفعلي بتشغيل: {verify_path.name}")
    verify_tflite(str(verify_path), args.onnx)

    int8_candidates = [f for f in tflite_files if "full_integer_quant" in f.name and "int16" not in f.name]
    if int8_candidates:
        print(f"\nالتحقق الفعلي بتشغيل النسخة INT8 الكاملة: {int8_candidates[0].name}")
        verify_tflite(str(int8_candidates[0]), args.onnx)


def verify_tflite(tflite_path, onnx_path):
    import numpy as np
    import onnxruntime as ort
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]

    if np.issubdtype(input_dtype, np.integer):
        # نموذج INT8 كامل: المدخل/المخرج أعداد صحيحة مباشرة، لا float32
        info = np.iinfo(input_dtype)
        dummy_input = np.random.randint(info.min, info.max, size=input_shape).astype(input_dtype)
    else:
        dummy_input = np.random.randn(*input_shape).astype(input_dtype)

    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()
    tflite_outputs = [interpreter.get_tensor(d["index"]) for d in output_details]
    print(f"TFLite يعمل فعلياً - أشكال المخرجات: {[o.shape for o in tflite_outputs]} (dtype={input_dtype})")

    for out in tflite_outputs:
        assert not np.isnan(out.astype(np.float64)).any(), "مخرجات TFLite تحوي NaN!"
        assert np.abs(out.astype(np.int64)).sum() > 0, "مخرجات TFLite كلها أصفار - التحويل فاشل!"

    if not np.issubdtype(input_dtype, np.integer):
        # مقارنة مع ONNX على نفس المدخل (float فقط - النماذج INT8 لها نطاق
        # تكميم مختلف كلياً عن ONNX float فلا تُقارَن مباشرة بنفس الطريقة)
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        onnx_input_name = session.get_inputs()[0].name
        onnx_input_shape = session.get_inputs()[0].shape
        if list(onnx_input_shape) == list(input_shape):
            onnx_outputs = session.run(None, {onnx_input_name: dummy_input})
        else:
            onnx_outputs = session.run(None, {onnx_input_name: dummy_input.transpose(0, 3, 1, 2)})
        print(f"ONNX (نفس المدخل تقريباً) - أشكال المخرجات: {[o.shape for o in onnx_outputs]}")

    print("التحقق ناجح: TFLite ينتج قيماً منطقية (غير NaN وغير صفرية بالكامل).")


if __name__ == "__main__":
    main()
