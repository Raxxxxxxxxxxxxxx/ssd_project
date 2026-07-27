import torch
import torch.nn as nn
from backbone import MobileNetV3Backbone
from utils import LAYER_NUM_ANCHORS, generate_anchors

# دقة الإدخال والطبقات المفعّلة لكل وضع تشغيل (ARM + DFMS، القسمان 3.4 و3.5)
MODE_CONFIG = {
    "TURBO":   {"resolution": (320, 320)},
    "NORMAL":  {"resolution": (300, 300)},
    "ECONOMY": {"resolution": (224, 224)},
}


class AdaptiveSSD(nn.Module):
    def __init__(self, num_classes, pretrained_backbone=True, anchor_wh=None):
        """
        anchor_wh: قاموس اختياري {"c1": [(w,h),...], ...} من
            utils.compute_layer_anchor_wh() (K-Means على بيانات حقيقية -
            القسم 3.7). إن تُرك None تُستخدم الصيغة الثابتة الافتراضية.
            يجب أن يُحفظ ويُحمَّل دائماً مع أوزان النموذج المدرَّبة عليه،
            لأن رؤوس التوقع (loc_heads) تتعلم إزاحات نسبية لهذه الأبعاد
            تحديداً - تغييرها بعد التدريب يُبطل معنى تلك الإزاحات.
        """
        super(AdaptiveSSD, self).__init__()
        self.num_classes = num_classes
        self.anchor_wh = anchor_wh

        # 1. استدعاء الـ Backbone (يعيد قائمة (اسم_الطبقة, feature_map) حسب الوضع)
        self.backbone = MobileNetV3Backbone(pretrained=pretrained_backbone)
        channels = self.backbone.out_channels  # {"c1":24, "c2":48, "c3":576, "c4":256}

        # 2. طبقات تحديد المواقع (Localization) والتصنيف (Classification)
        # كل طبقة لها عدد Anchors خاص بها (k) محدد في utils.LAYER_NUM_ANCHORS
        # حتى تبقى المقاييس متسقة بغض النظر عن الطبقات المفعّلة في الوضع الحالي.
        self.loc_heads = nn.ModuleDict()
        self.cls_heads = nn.ModuleDict()
        for name, in_ch in channels.items():
            k = LAYER_NUM_ANCHORS[name]
            self.loc_heads[name] = nn.Conv2d(in_ch, k * 4, kernel_size=3, padding=1)
            self.cls_heads[name] = nn.Conv2d(in_ch, k * num_classes, kernel_size=3, padding=1)

        # كاش لصناديق الإرساء لكل (وضع, أبعاد فعلية) لتفادي إعادة توليدها كل فريم
        self._anchor_cache = {}

    def forward(self, x, mode="NORMAL"):
        """
        يمرر الصورة ويقوم بالتوقع بناءً على الوضع الحالي (TURBO/NORMAL/ECONOMY)
        الذي تحدده وحدة DIC. يعيد (loc_outputs, cls_outputs, anchors).
        """
        features = self.backbone(x, mode=mode)  # [(name, feat), ...]

        loc_preds, cls_preds, layer_shapes = [], [], []
        for name, feat in features:
            loc = self.loc_heads[name](feat).permute(0, 2, 3, 1).contiguous()
            cls = self.cls_heads[name](feat).permute(0, 2, 3, 1).contiguous()
            loc_preds.append(loc.view(loc.size(0), -1, 4))
            cls_preds.append(cls.view(cls.size(0), -1, self.num_classes))
            layer_shapes.append((name, feat.shape[2], feat.shape[3]))

        loc_outputs = torch.cat(loc_preds, dim=1)
        cls_outputs = torch.cat(cls_preds, dim=1)

        anchors = self._get_anchors(mode, tuple(layer_shapes), device=x.device)

        return loc_outputs, cls_outputs, anchors

    def _get_anchors(self, mode, layer_shapes, device):
        """يولّد anchors مرة واحدة فقط لكل تركيبة (وضع + أبعاد) ثم يخزّنها مؤقتاً."""
        cache_key = (mode, layer_shapes)
        if cache_key not in self._anchor_cache:
            self._anchor_cache[cache_key] = generate_anchors(layer_shapes, custom_wh=self.anchor_wh)
        return self._anchor_cache[cache_key].to(device)


# اختبار النموذج للتأكد من ديناميكية الأوضاع بعد الإصلاح
if __name__ == "__main__":
    model = AdaptiveSSD(num_classes=21)
    model.eval()

    for mode in ("ECONOMY", "NORMAL", "TURBO"):
        size = MODE_CONFIG[mode]["resolution"]
        dummy = torch.randn(1, 3, size[1], size[0])
        with torch.no_grad():
            loc, cls, anchors = model(dummy, mode=mode)
        assert loc.shape[1] == anchors.shape[0], "عدد التوقعات يجب أن يطابق عدد الـ anchors"
        print(f"[{mode}] boxes={loc.shape}, classes={cls.shape}, anchors={anchors.shape}")
