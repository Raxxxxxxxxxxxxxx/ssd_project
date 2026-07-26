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
import os
import random
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou

from assd_model import AdaptiveSSD, MODE_CONFIG
from utils import VOC_CLASSES, preprocess, encode, center_to_corner

CLASS_TO_IDX = {name: idx for idx, name in enumerate(VOC_CLASSES)}


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

        import cv2
        image_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        frame = cv2.imread(image_path)
        h, w = frame.shape[:2]

        boxes, labels = self._parse_annotation(image_id, w, h)
        return frame, boxes, labels

    def _parse_annotation(self, image_id, img_w, img_h):
        """يستخرج صناديق الحقيقة وأصنافها، مُطبَّعة بين 0 و1 (صيغة corner)."""
        ann_path = os.path.join(self.annotations_dir, f"{image_id}.xml")
        root = ET.parse(ann_path).getroot()

        boxes, labels = [], []
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

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def collate_fn(batch):
    """كل صورة قد تحوي عدداً مختلفاً من الأجسام، لذا نُبقي القوائم كما هي."""
    frames, boxes, labels = zip(*batch)
    return list(frames), list(boxes), list(labels)


def random_horizontal_flip(frame, boxes, p=0.5):
    """تقليب أفقي عشوائي (Data Augmentation - القسم 2.7) مع تصحيح الصناديق."""
    if random.random() < p and len(boxes) > 0:
        frame = frame[:, ::-1, :].copy()
        boxes = boxes.clone()
        boxes[:, [0, 2]] = 1.0 - boxes[:, [2, 0]]
    return frame, boxes


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


def train_one_epoch(model, loader, criterion, optimizer, device, modes):
    model.train()
    running_loss = 0.0

    for frames, boxes_list, labels_list in loader:
        mode = random.choice(modes)
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

        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser(description="تدريب A-SSD على بيانات Pascal VOC")
    parser.add_argument("--data-root", required=True, help="مسار VOCdevkit/VOC20xx")
    parser.add_argument("--split", default="train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--lr-decay-epochs", nargs="+", type=int, default=[80, 100])
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

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

    model = AdaptiveSSD(num_classes=len(VOC_CLASSES)).to(device)
    criterion = MultiBoxLoss(neg_pos_ratio=3)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_epochs, gamma=0.1)

    modes = ["ECONOMY", "NORMAL", "TURBO"]

    print(f"بدء التدريب على {device} لعدد {len(dataset)} صورة، {args.epochs} حقبة (epoch).")
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, loader, criterion, optimizer, device, modes)
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs} - loss: {avg_loss:.4f} - lr: {scheduler.get_last_lr()[0]:.6f}")

        ckpt_path = os.path.join(args.checkpoint_dir, "last.pth")
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_path)

    print(f"اكتمل التدريب. آخر نسخة محفوظة في: {os.path.join(args.checkpoint_dir, 'last.pth')}")


if __name__ == "__main__":
    main()
