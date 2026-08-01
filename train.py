"""
حلقة تدريب A-SSD على بيانات بصيغة Pascal VOC (القسم 2.7 وفصل 4 من التقرير).

يتوقع بنية مجلدات VOC القياسية:
    data_root/
        JPEGImages/*.jpg
        Annotations/*.xml
        ImageSets/Main/{train,val}.txt

الاستخدام:
    python train.py --data-root /path/to/VOCdevkit/VOC2012 --epochs 100

ميزة مهمة: نظراً لأن الـ backbone ورؤوس التوقع (loc/cls heads) مشتركة بين
جميع الأوضاع الثلاثة (ECONOMY/NORMAL/TURBO) في AdaptiveSSD، تُختار عينة
عشوائية من الأوضاع الثلاثة لكل دفعة (batch) أثناء التدريب، بحيث يتعلم
النموذج جميع الدقات وجميع تركيبات الطبقات دفعة واحدة بدل تدريب ثلاث نماذج
منفصلة.
"""

import argparse
import itertools
import os
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.ops import box_iou

from assd_model import AdaptiveSSD, MODE_CONFIG
from utils import VOC_CLASSES, preprocess, encode, center_to_corner, compute_layer_anchor_wh

# فهرسة بالأصناف الحقيقية العشرين فقط (0..19)، دون "background"، لأن match_anchors()
# وevaluate.py يضيفان +1 لاحقاً لحجز الفهرس 0 للخلفية في مخرجات النموذج. تضمين
# "background" هنا كان يُنتج فهرسة مُزاحة مسبقاً (1..20) فيتكرر الإزاحة مرتين
# ويُخرج فهرساً خارج الحدود (21) عند "tvmonitor" أثناء حساب MultiBoxLoss.
CLASS_TO_IDX = {name: idx for idx, name in enumerate(VOC_CLASSES[1:])}


class VOCDataset(Dataset):
    """يقرأ صور وتعليقات (annotations) بصيغة Pascal VOC القياسية."""

    def __init__(self, data_root, split="train"):
        self.data_root = data_root
        self.images_dir = os.path.join(data_root, "JPEGImages")
        self.annotations_dir = os.path.join(data_root, "Annotations")
        split_file = os.path.join(data_root, "ImageSets", "Main", f"{split}.txt")

        with open(split_file) as f:
            self.ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]

        image_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        frame = cv2.imread(image_path)
        h, w = frame.shape[:2]

        boxes, labels, difficult = self._parse_annotation(image_id, w, h)
        return frame, boxes, labels, difficult

    def _parse_annotation(self, image_id, img_w, img_h):
        """يستخرج صناديق الحقيقة وأصنافها وعلامة difficult، مُطبَّعة بين 0 و1 (صيغة corner).
        علامة difficult تُستخدم فقط لاستثناء الجسم من حساب mAP في evaluate.py
        (بروتوكول VOC القياسي) - التدريب يستخدم كل الأجسام بلا استثناء."""
        ann_path = os.path.join(self.annotations_dir, f"{image_id}.xml")
        root = ET.parse(ann_path).getroot()

        boxes, labels, difficult = [], [], []
        for obj in root.findall("object"):
            name = obj.find("name").text.strip().lower()
            if name not in CLASS_TO_IDX:
                continue
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text) / img_w
            ymin = float(bbox.find("ymin").text) / img_h
            xmax = float(bbox.find("xmax").text) / img_w
            ymax = float(bbox.find("ymax").text) / img_h
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_TO_IDX[name])
            diff_tag = obj.find("difficult")
            difficult.append(int(diff_tag.text) if diff_tag is not None else 0)

        return (torch.tensor(boxes, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.long),
                torch.tensor(difficult, dtype=torch.long))


def collate_fn(batch):
    """كل صورة قد تحوي عدداً مختلفاً من الأجسام، لذا نُبقي القوائم كما هي."""
    frames, boxes, labels, difficult = zip(*batch)
    return list(frames), list(boxes), list(labels), list(difficult)


def random_horizontal_flip(frame, boxes, p=0.5):
    """تقليب أفقي عشوائي (Data Augmentation - القسم 2.7) مع تصحيح الصناديق."""
    if random.random() < p and len(boxes) > 0:
        frame = frame[:, ::-1, :].copy()
        boxes = boxes.clone()
        boxes[:, [0, 2]] = 1.0 - boxes[:, [2, 0]]
    return frame, boxes


def random_color_jitter(frame, p=0.5):
    """تشويش لوني عشوائي (سطوع/تباين/تشبع) - القسم 2.7. لا يغيّر الصناديق
    إطلاقاً لأنه يعمل على قيم البكسل فقط."""
    if random.random() >= p:
        return frame

    frame = frame.astype(np.float32)

    if random.random() < 0.5:  # سطوع (Brightness)
        frame += random.uniform(-32, 32)

    if random.random() < 0.5:  # تباين (Contrast)
        frame = 127.5 + (frame - 127.5) * random.uniform(0.7, 1.3)

    frame = np.clip(frame, 0, 255).astype(np.uint8)

    if random.random() < 0.5:  # تشبع (Saturation) عبر HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= random.uniform(0.7, 1.3)
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return frame


def random_crop(frame, boxes, labels, min_scale=0.3, max_trials=20):
    """اقتصاص عشوائي (Random Crop) مع تصحيح صناديق الحقيقة - القسم 2.7.
    يُبقي فقط الصناديق التي يقع مركزها داخل منطقة الاقتصاص، ويعيد الصورة
    الأصلية دون تعديل إذا فشلت كل المحاولات أو لم توجد صناديق أصلاً."""
    if boxes.numel() == 0:
        return frame, boxes, labels

    h, w = frame.shape[:2]
    boxes_np = boxes.numpy()

    for _ in range(max_trials):
        scale = random.uniform(min_scale, 1.0)
        aspect = random.uniform(0.5, 2.0)
        crop_w = min(int(w * scale * (aspect ** 0.5)), w)
        crop_h = min(int(h * scale / (aspect ** 0.5)), h)
        if crop_w < 10 or crop_h < 10:
            continue

        x1 = random.randint(0, w - crop_w)
        y1 = random.randint(0, h - crop_h)
        x2, y2 = x1 + crop_w, y1 + crop_h

        cx = (boxes_np[:, 0] + boxes_np[:, 2]) / 2 * w
        cy = (boxes_np[:, 1] + boxes_np[:, 3]) / 2 * h
        keep = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
        if not keep.any():
            continue

        cropped_frame = frame[y1:y2, x1:x2]
        kept = boxes_np[keep]
        new_boxes = np.empty_like(kept)
        new_boxes[:, 0] = np.clip((kept[:, 0] * w - x1) / crop_w, 0, 1)
        new_boxes[:, 1] = np.clip((kept[:, 1] * h - y1) / crop_h, 0, 1)
        new_boxes[:, 2] = np.clip((kept[:, 2] * w - x1) / crop_w, 0, 1)
        new_boxes[:, 3] = np.clip((kept[:, 3] * h - y1) / crop_h, 0, 1)

        kept_labels = labels[torch.from_numpy(keep)]
        return cropped_frame, torch.from_numpy(new_boxes).float(), kept_labels

    return frame, boxes, labels


def match_anchors(anchors, gt_boxes, gt_labels, iou_threshold=0.5):
    """
    يطابق صناديق الحقيقة (ground truth) مع الـ anchors حسب IoU (القسم 2.7):
    - لكل anchor: أفضل صندوق حقيقة متطابق (إن وُجد IoU >= 0.5) وإلا فهو خلفية (0).
    - يضمن أيضاً أن كل صندوق حقيقة يُطابق مع الأنسب له من الـ anchors على الأقل مرة.
    يعيد: (loc_targets مرمّزة بصيغة SSD، cls_targets لكل anchor).
    """
    anchors_corner = center_to_corner(anchors)
    num_anchors = anchors.size(0)

    if gt_boxes.numel() == 0:
        return torch.zeros(num_anchors, 4), torch.zeros(num_anchors, dtype=torch.long)

    iou = box_iou(gt_boxes, anchors_corner)  # (num_gt, num_anchors)

    best_gt_iou, best_gt_idx = iou.max(dim=0)  # لكل anchor: أفضل gt
    best_anchor_iou, best_anchor_idx = iou.max(dim=1)  # لكل gt: أفضل anchor

    # نضمن أن كل gt يُطابق مع الأنسب له من الـ anchors حتى لو كان IoU منخفضاً
    best_gt_iou[best_anchor_idx] = 1.0
    for gt_i, anchor_i in enumerate(best_anchor_idx):
        best_gt_idx[anchor_i] = gt_i

    matched_boxes = gt_boxes[best_gt_idx]
    cls_targets = gt_labels[best_gt_idx] + 1  # +1 لأن 0 محجوزة للخلفية
    cls_targets[best_gt_iou < iou_threshold] = 0  # خلفية

    loc_targets = encode(matched_boxes, anchors)
    return loc_targets, cls_targets


class MultiBoxLoss(nn.Module):
    """دالة الخسارة المركّبة (القسم 2.6): Smooth L1 للمواقع + Cross-Entropy
    للتصنيف مع Hard Negative Mining بنسبة سلبي:إيجابي = 3:1."""

    def __init__(self, neg_pos_ratio=3):
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, loc_preds, cls_preds, loc_targets, cls_targets):
        pos_mask = cls_targets > 0
        num_pos = pos_mask.sum(dim=1, keepdim=True).clamp(min=1)

        # 1. خسارة المواقع (Smooth L1) على الـ anchors الإيجابية فقط
        loc_loss = F.smooth_l1_loss(
            loc_preds[pos_mask], loc_targets[pos_mask], reduction="sum"
        )

        # 2. خسارة التصنيف مع Hard Negative Mining
        batch_size, num_anchors, num_classes = cls_preds.shape
        cls_loss_all = F.cross_entropy(
            cls_preds.view(-1, num_classes), cls_targets.view(-1), reduction="none"
        ).view(batch_size, num_anchors)

        cls_loss_pos = (cls_loss_all * pos_mask.float()).sum()

        # اختيار أصعب الأمثلة السلبية فقط (أعلى خسارة) بنسبة 1:3
        neg_loss_view = cls_loss_all.clone()
        neg_loss_view[pos_mask] = -1.0  # نستبعد الإيجابية من الترتيب
        num_neg = (num_pos.squeeze(1) * self.neg_pos_ratio).long().clamp(max=num_anchors - 1)

        _, ranked_idx = neg_loss_view.sort(dim=1, descending=True)
        rank = torch.zeros_like(ranked_idx)
        rank.scatter_(1, ranked_idx, torch.arange(num_anchors, device=cls_preds.device).expand_as(ranked_idx))
        neg_mask = rank < num_neg.unsqueeze(1)

        cls_loss_neg = (cls_loss_all * neg_mask.float()).sum()

        total_pos = num_pos.sum().clamp(min=1)
        loc_loss = loc_loss / total_pos
        cls_loss = (cls_loss_pos + cls_loss_neg) / total_pos

        return loc_loss + cls_loss, loc_loss.item(), cls_loss.item()


def train_one_epoch(model, loader, criterion, optimizer, device, mode_cycle):
    """يعيد قاموس خسارة منفصل لكل وضع {"TURBO":..,"NORMAL":..,"ECONOMY":..,"all":..}
    بدل رقم واحد مجمّع، حتى يظهر إن كان وضع معيّن يتعلّم أبطأ من غيره."""
    model.train()
    running_loss = {"TURBO": 0.0, "NORMAL": 0.0, "ECONOMY": 0.0}
    batch_count = {"TURBO": 0, "NORMAL": 0, "ECONOMY": 0}

    for frames, boxes_list, labels_list, _ in loader:
        aug_frames, aug_boxes, aug_labels = [], [], []
        for frame, boxes, labels in zip(frames, boxes_list, labels_list):
            frame, boxes, labels = random_crop(frame, boxes, labels)
            frame, boxes = random_horizontal_flip(frame, boxes)
            frame = random_color_jitter(frame)
            aug_frames.append(frame)
            aug_boxes.append(boxes)
            aug_labels.append(labels)
        frames, boxes_list, labels_list = aug_frames, aug_boxes, aug_labels

        # round-robin بدل اختيار عشوائي: يضمن أن كل وضع (TURBO/NORMAL/ECONOMY)
        # يتلقى بالضبط ثلث الدُفعات (batches) عبر الحقبة كاملة، بدل تفاوت
        # عشوائي قد يُحرم بعض الأوضاع من نصيبها العادل من إشارة التدريب.
        mode = next(mode_cycle)
        resolution = MODE_CONFIG[mode]["resolution"]

        batch_tensors = torch.cat([preprocess(f, resolution) for f in frames], dim=0).to(device)

        optimizer.zero_grad()
        loc_preds, cls_preds, anchors = model(batch_tensors, mode=mode)
        anchors = anchors.to(device)

        loc_targets_batch, cls_targets_batch = [], []
        for boxes, labels in zip(boxes_list, labels_list):
            boxes, labels = boxes.to(device), labels.to(device)
            loc_t, cls_t = match_anchors(anchors, boxes, labels)
            loc_targets_batch.append(loc_t)
            cls_targets_batch.append(cls_t)

        loc_targets = torch.stack(loc_targets_batch).to(device)
        cls_targets = torch.stack(cls_targets_batch).to(device)

        loss, loc_l, cls_l = criterion(loc_preds, cls_preds, loc_targets, cls_targets)
        loss.backward()
        optimizer.step()

        running_loss[mode] += loss.item()
        batch_count[mode] += 1

    avg_loss = {m: running_loss[m] / max(batch_count[m], 1) for m in running_loss}
    total_batches = sum(batch_count.values())
    avg_loss["all"] = sum(running_loss.values()) / max(total_batches, 1)
    return avg_loss


@torch.no_grad()
def _quick_eval(model, val_dataset, device, mode="NORMAL"):
    """تقييم mAP@0.5 سريع بالنموذج الحالي في الذاكرة (بلا إعادة تحميل من قرص)،
    يُستخدم للتقييم الدوري أثناء التدريب (--eval-every) ولتتبّع أفضل نسخة."""
    from evaluate import evaluate as run_eval
    from assd_model import MODE_CONFIG as _MODE_CONFIG
    from utils import postprocess

    was_training = model.training
    model.eval()
    resolution = _MODE_CONFIG[mode]["resolution"]

    def detect_fn(frame):
        h, w = frame.shape[:2]
        input_tensor = preprocess(frame, resolution).to(device)
        loc_preds, cls_preds, anchors = model(input_tensor, mode=mode)
        return postprocess(loc_preds, cls_preds, anchors, orig_size=(w, h),
                            conf_thresh=0.01, nms_thresh=0.45, top_k=200)

    aps = run_eval(val_dataset, detect_fn)
    if was_training:
        model.train()
    return sum(aps.values()) / max(len(aps), 1)


def main():
    parser = argparse.ArgumentParser(description="تدريب A-SSD على بيانات Pascal VOC")
    parser.add_argument("--data-root", required=True, help="مسار VOCdevkit/VOC20xx")
    parser.add_argument("--split", default="train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--lr-decay-epochs", nargs="+", type=int, default=None,
                         help="حقب خفض معدل التعلّم. افتراضياً (إن لم تُحدَّد) تُشتق تلقائياً "
                              "كنسبة 70%% و90%% من --epochs، حتى لا تفقد القيمة الافتراضية "
                              "الثابتة [80,100] أثرها عند تغيير --epochs (مثلاً لجولات ablation قصيرة).")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                         help="عدد حقب الإحماء الخطي لمعدل التعلّم (LR warmup)")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", default=None, help="مسار checkpoint لاستئناف التدريب منه")
    parser.add_argument("--eval-every", type=int, default=10,
                         help="تقييم mAP دوري كل كم حقبة أثناء التدريب، مع حفظ أفضل نسخة "
                              "(best.pth). صفر لتعطيله.")
    parser.add_argument("--eval-split", default="val", help="اسم split للتقييم الدوري أثناء التدريب")
    parser.add_argument("--eval-subset", type=int, default=200,
                         help="عدد صور عشوائية (بذرة ثابتة) من eval-split للتقييم الدوري "
                              "السريع أثناء التدريب. صفر = المجموعة كاملة (أبطأ).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.lr_decay_epochs is None:
        args.lr_decay_epochs = [max(1, round(args.epochs * 0.7)), max(1, round(args.epochs * 0.9))]
        print(f"لم يُحدَّد --lr-decay-epochs، اشتُقّ تلقائياً من --epochs={args.epochs}: "
              f"{args.lr_decay_epochs} (يترك حقباً بعد آخر خفض للتقارب، خلافاً لـ[80,100] الثابت سابقاً).")

    if not os.path.isdir(args.data_root):
        raise SystemExit(
            f"مسار البيانات غير موجود: {args.data_root}\n"
            "نزّل Pascal VOC 2012 ثم مرّر --data-root إلى مجلد VOCdevkit/VOC2012."
        )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device(args.device)

    dataset = VOCDataset(args.data_root, split=args.split)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, drop_last=True,
    )

    val_dataset = None
    if args.eval_every > 0:
        try:
            full_val = VOCDataset(args.data_root, split=args.eval_split)
            if 0 < args.eval_subset < len(full_val):
                rng = random.Random(42)
                indices = rng.sample(range(len(full_val)), args.eval_subset)
                val_dataset = Subset(full_val, indices)
            else:
                val_dataset = full_val
            print(f"تقييم دوري مفعّل كل {args.eval_every} حقبة على {len(val_dataset)} "
                  f"صورة من split={args.eval_split} (أفضل نسخة تُحفظ في best.pth).")
        except FileNotFoundError:
            print(f"تحذير: split={args.eval_split} غير موجود - تعطيل التقييم الدوري وbest.pth.")
            args.eval_every = 0

    start_epoch = 1
    anchor_wh = None

    if args.resume:
        # عند الاستئناف نُعيد استخدام anchor_wh المحفوظة مع النموذج نفسه (إن
        # وُجدت)، لأن رؤوس التوقع تعلّمت إزاحات نسبية لهذه الأبعاد تحديداً -
        # حساب K-Means من جديد هنا قد يُنتج أبعاداً مختلفة ويُبطل التدريب السابق.
        ckpt = torch.load(args.resume, map_location=device)
        anchor_wh = ckpt.get("anchor_wh")
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"استئناف التدريب من {args.resume} عند الحقبة {start_epoch}.")
    else:
        print("حساب K-Means anchors من التوزيع الحقيقي لصناديق VOC2012 (القسم 3.7)...")
        anchor_wh = compute_layer_anchor_wh(dataset)
        for name, wh in anchor_wh.items():
            print(f"  {name}: {['(%.3f, %.3f)' % p for p in wh]}")

    model = AdaptiveSSD(num_classes=len(VOC_CLASSES), anchor_wh=anchor_wh).to(device)
    if args.resume:
        model.load_state_dict(ckpt["model"])

    criterion = MultiBoxLoss(neg_pos_ratio=3)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    if args.resume and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    # جدولة LR: إحماء خطي (warmup) لأول عدة حقب ثم خفض عند lr_decay_epochs،
    # كما في التقرير (قسم 2.7) مع إضافة الإحماء (ممارسة قياسية لتحسين
    # الاستقرار في أول الحقب، غير مذكورة بالتقرير لكنها لا تتعارض معه).
    warmup_epochs = max(0, args.warmup_epochs)
    decay_milestones = [max(1, m - warmup_epochs) for m in args.lr_decay_epochs]
    main_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_milestones, gamma=0.1)
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs]
        )
    else:
        scheduler = main_scheduler

    # نُنشئ الدوّار (cycle) مرة واحدة فقط ونمرّره لكل استدعاء train_one_epoch
    # حتى يستمر التوزيع المتساوي عبر حدود الحقب (لا يُعاد ضبطه كل حقبة).
    mode_cycle = itertools.cycle(["TURBO", "NORMAL", "ECONOMY"])

    best_map = -1.0
    print(f"بدء التدريب على {device} لعدد {len(dataset)} صورة، من الحقبة {start_epoch} حتى {args.epochs}.")
    for epoch in range(start_epoch, args.epochs + 1):
        avg_loss = train_one_epoch(model, loader, criterion, optimizer, device, mode_cycle)
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs} - loss(all): {avg_loss['all']:.4f} "
              f"[TURBO {avg_loss['TURBO']:.4f} | NORMAL {avg_loss['NORMAL']:.4f} | "
              f"ECONOMY {avg_loss['ECONOMY']:.4f}] - lr: {scheduler.get_last_lr()[0]:.6f}")

        ckpt_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "anchor_wh": anchor_wh,
        }
        torch.save(ckpt_data, os.path.join(args.checkpoint_dir, "last.pth"))

        # نسخة دورية كل 10 حقب (بالإضافة إلى last.pth) لتقييم التقدّم أثناء
        # تدريب طويل غير مراقب عبر evaluate.py دون انتظار الاكتمال الكامل.
        if epoch % 10 == 0 or epoch == args.epochs:
            torch.save(ckpt_data, os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pth"))

        if args.eval_every > 0 and (epoch % args.eval_every == 0 or epoch == args.epochs):
            mean_ap = _quick_eval(model, val_dataset, device, mode="NORMAL")
            print(f"  [تقييم دوري] mAP@0.5 (NORMAL, {len(val_dataset)} صورة) عند الحقبة {epoch}: {mean_ap * 100:.2f}%")
            if mean_ap > best_map:
                best_map = mean_ap
                torch.save(ckpt_data, os.path.join(args.checkpoint_dir, "best.pth"))
                print(f"  [أفضل نسخة جديدة] حُفظت في best.pth (mAP={mean_ap * 100:.2f}%)")

    print(f"اكتمل التدريب. آخر نسخة محفوظة في: {os.path.join(args.checkpoint_dir, 'last.pth')}")
    if best_map >= 0:
        print(f"أفضل نسخة أثناء التدريب: {os.path.join(args.checkpoint_dir, 'best.pth')} (mAP={best_map * 100:.2f}%)")


if __name__ == "__main__":
    main()
