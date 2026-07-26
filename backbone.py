import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV3Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV3Backbone, self).__init__()
        # 1. تحميل نموذج MobileNetV3-Small الممرن مسبقاً جاهزاً من PyTorch
        # هذا يمنحنا سرعة وخفة (حجم 5.8 ميغابايت فقط مقارنة بـ 528 ميغابايت لـ VGG16)
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base_model = models.mobilenet_v3_small(weights=weights)
        features = base_model.features

        # 2. تقسيم طبقات الشبكة لاستخراج ميزات بأبعاد مختلفة (Multi-scale Feature Maps)
        # نحتاج لاستخراج الطبقات في مراحل (Stages) مختلفة لتغذية نموذج الـ SSD
        self.stage1 = features[:4]   # C1: بديل Conv4_3 (عالي الدقة، يفيد الأجسام الصغيرة)
        self.stage2 = features[4:9]  # C2: بديل FC6/FC7
        self.stage3 = features[9:]   # C3: بديل Conv8_2 (دقة منخفضة، يفيد الأجسام الكبيرة)

        # 3. طبقة إضافية (Extra Layer) خاصة بوضع TURBO فقط، بديل Conv9_2
        # تستخرج ميزات C4 بدقة أخفض (stride ~64) لكشف أدق للأجسام الكبيرة
        self.extra = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.Hardswish(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.Hardswish(),
        )

        self.out_channels = {"c1": 24, "c2": 48, "c3": 576, "c4": 256}

    def forward(self, x, mode="NORMAL"):
        """
        يمرر الإدخال عبر الطبقات بالتوالي، ويعيد فقط الطبقات المفعّلة في
        الوضع الحالي حسب جدول DFMS الوارد في التقرير (القسم 3.5):
          TURBO   -> [C1, C2, C3, C4]  (4 طبقات)
          NORMAL  -> [C1, C2, C3]      (3 طبقات)
          ECONOMY -> [C2, C3]          (طبقتان فقط، تُهمل C1 توفيراً للحساب)
        الإخراج: قائمة من أزواج (اسم_الطبقة, feature_map) بالترتيب.
        """
        f1 = self.stage1(x)   # C1
        f2 = self.stage2(f1)  # C2
        f3 = self.stage3(f2)  # C3

        if mode == "TURBO":
            f4 = self.extra(f3)  # C4
            return [("c1", f1), ("c2", f2), ("c3", f3), ("c4", f4)]
        elif mode == "ECONOMY":
            return [("c2", f2), ("c3", f3)]
        else:  # NORMAL
            return [("c1", f1), ("c2", f2), ("c3", f3)]


# اختبار الكود للتأكد من أن الأبعاد تخرج بشكل صحيح
if __name__ == "__main__":
    model = MobileNetV3Backbone()
    for mode, size in [("ECONOMY", 224), ("NORMAL", 300), ("TURBO", 320)]:
        dummy_input = torch.randn(1, 3, size, size)
        feats = model(dummy_input, mode=mode)
        shapes = ", ".join(f"{name}={tuple(t.shape)}" for name, t in feats)
        print(f"[{mode}] input={size}x{size} -> {shapes}")
