import torch
from adaptive_engine import AdaptiveEngine
from assd_model import AdaptiveSSD
from utils import preprocess, postprocess


class AdaptiveInference:
    def __init__(self, num_classes=21, conf_thresh=0.5, nms_thresh=0.45,
                 checkpoint=None, pretrained_backbone=True):
        # 1. استدعاء محرك التكيف (DIC يعمل في خيط خلفي منفصل - غير حاجب)
        self.engine = AdaptiveEngine()

        # anchor_wh (إن وُجدت داخل الـcheckpoint) يجب تحميلها مع الأوزان
        # دائماً: رؤوس التوقع تعلّمت إزاحات نسبية لأبعاد anchors محدَّدة وقت
        # التدريب - استخدام أبعاد مختلفة الآن يُبطل معنى تلك الإزاحات.
        anchor_wh = None
        state_dict = None
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location="cpu")
            anchor_wh = ckpt.get("anchor_wh")
            state_dict = ckpt.get("model", ckpt)

        self.model = AdaptiveSSD(num_classes=num_classes, pretrained_backbone=pretrained_backbone,
                                  anchor_wh=anchor_wh)
        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        self.model.eval()  # وضع النموذج في طور الاستدلال (Inference) وليس التدريب
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

    def run_inference(self, frame):
        """
        يستقبل فريم (صورة BGR من OpenCV)، ويقوم بمعالجتها تكيفياً بناءً على
        آخر قرار اتخذه DIC، ثم يعيد اكتشافات نهائية جاهزة للرسم مباشرة.
        """
        # 2. قراءة آخر وضع (استدعاء غير حاجب، لا ينتظر أي قياس جديد)
        status = self.engine.get_current_mode()
        mode = status["mode"]
        resolution = status["resolution"]

        # 3. تجهيز الصورة: BGR->RGB + تغيير الأبعاد + تطبيع ImageNet
        input_tensor = preprocess(frame, resolution)
        orig_h, orig_w = frame.shape[:2]

        # 4. تشغيل التوقع عبر النموذج بالاعتماد على الوضع الحالي
        with torch.no_grad():
            loc_preds, cls_preds, anchors = self.model(input_tensor, mode=mode)
            detections = postprocess(
                loc_preds, cls_preds, anchors,
                orig_size=(orig_w, orig_h),
                conf_thresh=self.conf_thresh,
                nms_thresh=self.nms_thresh,
            )

        return detections, status


# تجربة تشغيل النظام المتكامل بالكامل
if __name__ == "__main__":
    import numpy as np

    inference_system = AdaptiveInference()

    # محاكاة فريم وهمي قادم من الكاميرا بأبعاد (640x480)
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    detections, system_status = inference_system.run_inference(dummy_frame)

    print("Full System Inference Completed Successfully!")
    print(f"Mode: {system_status['mode']} | Resolution: {system_status['resolution']}")
    print(f"Detections found: {len(detections)} (نموذج غير مدرَّب بعد، فالنتائج عشوائية)")
