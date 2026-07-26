"""
أدوات مساعدة لخوارزمية A-SSD:
- توليد صناديق الإرساء (Anchor Boxes) لكل طبقة feature map حسب الوضع الحالي.
- ترميز/فك ترميز الصناديق (Encode/Decode) بأسلوب SSD القياسي.
- Non-Maximum Suppression لإزالة التكرار.
- ما قبل/بعد المعالجة (preprocess/postprocess) لربط الكاميرا بالنموذج.
- K-Means لتحسين مقاسات الـ Anchors من بيانات حقيقية (قسم 3.7 من التقرير).
"""

import math
import cv2
import numpy as np
import torch
from torchvision.ops import nms as torch_nms

# إحصائيات تطبيع ImageNet التي دُرِّب عليها MobileNetV3-Small مسبقاً
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# متغيرات الترميز القياسية في SSD لتحويل إزاحات الشبكة إلى إحداثيات فعلية
VARIANCES = (0.1, 0.1, 0.2, 0.2)

# إعداد كل طبقة: k = عدد الـ Anchors في كل موقع، scale/next_scale لحساب الأبعاد،
# ratios = نسب الأبعاد المستخدمة. هذا الترتيب ثابت بغض النظر عن عدد الطبقات
# المفعّلة في الوضع الحالي (ECONOMY/NORMAL/TURBO) حتى تبقى المقاييس متسقة.
LAYER_CONFIG = {
    "c1": {"scale": 0.20, "next_scale": 0.433, "ratios": [1.0, 2.0, 0.5]},
    "c2": {"scale": 0.433, "next_scale": 0.667, "ratios": [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0]},
    "c3": {"scale": 0.667, "next_scale": 0.90, "ratios": [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0]},
    "c4": {"scale": 0.90, "next_scale": 1.05, "ratios": [1.0, 2.0, 0.5]},
}

# عدد الـ Anchors لكل موقع في كل طبقة (= عدد النسب + 1 صندوق إضافي بمقياس وسطي)
LAYER_NUM_ANCHORS = {name: len(cfg["ratios"]) + 1 for name, cfg in LAYER_CONFIG.items()}


def generate_anchors(layer_shapes):
    """
    يولّد صناديق الإرساء لمجموعة من طبقات الـ feature maps.
    layer_shapes: قائمة من (اسم_الطبقة, ارتفاع, عرض) بحسب الطبقات المفعّلة فعلياً.
    الإخراج: Tensor بشكل (N, 4) بصيغة (cx, cy, w, h) مُطبَّعة بين 0 و1.
    """
    anchors = []
    for name, h, w in layer_shapes:
        cfg = LAYER_CONFIG[name]
        sk = cfg["scale"]
        sk_next = cfg["next_scale"]
        extra_box_size = math.sqrt(sk * sk_next)
        for i in range(h):
            cy = (i + 0.5) / h
            for j in range(w):
                cx = (j + 0.5) / w
                for ratio in cfg["ratios"]:
                    box_w = sk * math.sqrt(ratio)
                    box_h = sk / math.sqrt(ratio)
                    anchors.append((cx, cy, box_w, box_h))
                # صندوق إضافي مربع بمقياس هندسي وسطي بين هذه الطبقة والتالية
                anchors.append((cx, cy, extra_box_size, extra_box_size))

    anchors_t = torch.tensor(anchors, dtype=torch.float32)
    return anchors_t.clamp(0.0, 1.0)


def center_to_corner(boxes):
    """تحويل (cx, cy, w, h) إلى (xmin, ymin, xmax, ymax)."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def corner_to_center(boxes):
    """تحويل (xmin, ymin, xmax, ymax) إلى (cx, cy, w, h)."""
    xmin, ymin, xmax, ymax = boxes.unbind(-1)
    return torch.stack([(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin], dim=-1)


def encode(matched_boxes, anchors, variances=VARIANCES):
    """
    يحوّل صناديق الحقيقة المطابقة (ground-truth بصيغة corner) إلى إزاحات
    نسبية للـ anchors، تُستخدم كأهداف (targets) أثناء التدريب.
    """
    matched_center = corner_to_center(matched_boxes)
    g_cxcy = (matched_center[..., :2] - anchors[..., :2]) / (variances[0] * anchors[..., 2:])
    g_wh = torch.log(matched_center[..., 2:].clamp(min=1e-6) / anchors[..., 2:]) / variances[2]
    return torch.cat([g_cxcy, g_wh], dim=-1)


def decode(loc_preds, anchors, variances=VARIANCES):
    """
    يفكّ ترميز مخرجات الشبكة (loc_preds) بالاستناد إلى الـ anchors المرجعية،
    ويعيد الصناديق النهائية بصيغة (xmin, ymin, xmax, ymax) مُطبَّعة [0,1].
    """
    cxcy = anchors[..., :2] + loc_preds[..., :2] * variances[0] * anchors[..., 2:]
    wh = anchors[..., 2:] * torch.exp(loc_preds[..., 2:] * variances[2])
    boxes = torch.cat([cxcy, wh], dim=-1)
    return center_to_corner(boxes).clamp(0.0, 1.0)


def preprocess(frame_bgr, resolution):
    """
    يجهّز فريم OpenCV (BGR) ليصبح Tensor متوافق مع MobileNetV3 المُدرَّب على
    ImageNet: تحويل BGR->RGB، تغيير الأبعاد، تطبيع بالـ mean/std القياسيين.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, resolution, interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor.unsqueeze(0)


def postprocess(loc_preds, cls_preds, anchors, orig_size,
                conf_thresh=0.5, nms_thresh=0.45, top_k=200):
    """
    يحوّل مخرجات النموذج الخام إلى قائمة اكتشافات نهائية جاهزة للرسم:
    [(xmin, ymin, xmax, ymax) بالبكسل, class_id, score), ...]
    orig_size: (width, height) للصورة الأصلية قبل تغيير الحجم.
    """
    orig_w, orig_h = orig_size
    boxes_norm = decode(loc_preds[0], anchors)
    scores_all = torch.softmax(cls_preds[0], dim=-1)

    num_classes = scores_all.shape[-1]
    detections = []

    for class_id in range(1, num_classes):  # نتجاوز الخلفية (background = 0)
        class_scores = scores_all[:, class_id]
        mask = class_scores > conf_thresh
        if mask.sum() == 0:
            continue

        boxes = boxes_norm[mask]
        scores = class_scores[mask]

        keep = torch_nms(boxes, scores, nms_thresh)
        keep = keep[:top_k]

        for idx in keep:
            xmin, ymin, xmax, ymax = boxes[idx].tolist()
            pixel_box = (
                int(xmin * orig_w), int(ymin * orig_h),
                int(xmax * orig_w), int(ymax * orig_h),
            )
            detections.append((pixel_box, class_id, float(scores[idx])))

    detections.sort(key=lambda d: d[2], reverse=True)
    return detections[:top_k]


def kmeans_anchor_boxes(boxes_wh, k, iters=100, seed=42):
    """
    يحسب k أبعاد Anchor مثلى من صناديق حقيقة (ground-truth) بأسلوب K-Means
    باستخدام مقياس (1 - IoU) كمسافة، كما ورد في القسم 3.7 من التقرير.
    boxes_wh: مصفوفة numpy بشكل (N, 2) تحوي (width, height) مُطبَّعة [0,1].
    يُعاد: مصفوفة (k, 2) بأبعاد الـ Anchors المختارة.
    """
    rng = np.random.default_rng(seed)
    boxes_wh = np.asarray(boxes_wh, dtype=np.float64)
    n = boxes_wh.shape[0]
    if n == 0:
        raise ValueError("لا توجد صناديق لحساب K-Means عليها")

    clusters = boxes_wh[rng.choice(n, size=min(k, n), replace=False)]

    def iou(box, clusters):
        w = np.minimum(box[0], clusters[:, 0])
        h = np.minimum(box[1], clusters[:, 1])
        intersection = w * h
        box_area = box[0] * box[1]
        cluster_area = clusters[:, 0] * clusters[:, 1]
        return intersection / (box_area + cluster_area - intersection + 1e-9)

    last_assignments = np.zeros(n)
    for _ in range(iters):
        distances = 1 - np.array([iou(box, clusters) for box in boxes_wh])
        assignments = distances.argmin(axis=1)

        if (assignments == last_assignments).all():
            break
        last_assignments = assignments

        for cluster_idx in range(clusters.shape[0]):
            members = boxes_wh[assignments == cluster_idx]
            if len(members) > 0:
                clusters[cluster_idx] = members.mean(axis=0)

    return clusters


VOC_CLASSES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
