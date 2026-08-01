#!/usr/bin/env bash
# يحمّل VOC2007 trainval + VOC2007 test + VOC2012 trainval، ويوحّدهم في جذر
# بيانات واحد بثلاث قوائم انقسام (splits):
#   train.txt         -> VOC2007 trainval + VOC2012 trainval (16,551 صورة) - للتدريب
#   voc2007test.txt    -> VOC2007 test فقط (4,952 صورة) - لمقارنة عادلة مع
#                          رقم الأطروحة المرجعي 71.4%/74.3% (نفس بروتوكول SSD الأصلي)
#
# لا تصادم في المعرّفات (IDs) بين 2007 (أرقام مجردة مثل 000005) و2012
# (بادئة سنة مثل 2008_000008) - الدمج المباشر آمن ومطابق لبروتوكول "07+12"
# القياسي المستخدم في كل أوراق SSD/Faster-RCNN تقريباً.
#
# الاستخدام (على Colab/Kaggle، أو أي جهاز فيه مساحة قرص ~3GB واتصال إنترنت):
#   bash prepare_voc0712.sh /content/VOC0712
#
# بعدها للتدريب:
#   python train.py --data-root /content/VOC0712 --split train \
#       --eval-split voc2007test --epochs 100 --batch-size 32 --device cuda \
#       --checkpoint-dir /content/drive/MyDrive/assd_checkpoints_0712
#
# وللتقييم النهائي بنفس بروتوكول الأطروحة (VOC2007 test):
#   python evaluate.py --data-root /content/VOC0712 --split voc2007test \
#       --checkpoint /content/drive/MyDrive/assd_checkpoints_0712/best.pth \
#       --mode NORMAL --by-size

set -euo pipefail

OUT_ROOT="${1:-./VOC0712}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$OUT_ROOT/JPEGImages" "$OUT_ROOT/Annotations" "$OUT_ROOT/ImageSets/Main"

download() {
    # يجرّب المصدر الرسمي أولاً، ثم مرآة pjreddie المعروفة إن فشل (الخادم
    # الرسمي لأكسفورد يتعطّل بشكل متكرر ومعروف في المجتمع).
    local primary="$1" mirror="$2" out="$3"
    echo "تحميل: $out"
    if ! curl -fL --retry 2 -o "$out" "$primary"; then
        echo "  فشل المصدر الرسمي، تجربة المرآة..."
        curl -fL --retry 2 -o "$out" "$mirror"
    fi
}

download \
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar" \
    "https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar" \
    "$TMP_DIR/VOCtrainval_2007.tar"

download \
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar" \
    "https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar" \
    "$TMP_DIR/VOCtest_2007.tar"

download \
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar" \
    "https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar" \
    "$TMP_DIR/VOCtrainval_2012.tar"

echo "فك الضغط..."
mkdir -p "$TMP_DIR/2007trainval" "$TMP_DIR/2007test" "$TMP_DIR/2012trainval"
tar -xf "$TMP_DIR/VOCtrainval_2007.tar" -C "$TMP_DIR/2007trainval"
tar -xf "$TMP_DIR/VOCtest_2007.tar" -C "$TMP_DIR/2007test"
tar -xf "$TMP_DIR/VOCtrainval_2012.tar" -C "$TMP_DIR/2012trainval"

V2007TV="$TMP_DIR/2007trainval/VOCdevkit/VOC2007"
V2007TE="$TMP_DIR/2007test/VOCdevkit/VOC2007"
V2012TV="$TMP_DIR/2012trainval/VOCdevkit/VOC2012"

echo "دمج الصور والتعليقات في $OUT_ROOT ..."
cp "$V2007TV"/JPEGImages/*.jpg "$OUT_ROOT/JPEGImages/"
cp "$V2007TV"/Annotations/*.xml "$OUT_ROOT/Annotations/"
cp "$V2007TE"/JPEGImages/*.jpg "$OUT_ROOT/JPEGImages/"
cp "$V2007TE"/Annotations/*.xml "$OUT_ROOT/Annotations/"
cp "$V2012TV"/JPEGImages/*.jpg "$OUT_ROOT/JPEGImages/"
cp "$V2012TV"/Annotations/*.xml "$OUT_ROOT/Annotations/"

cat "$V2007TV/ImageSets/Main/trainval.txt" "$V2012TV/ImageSets/Main/trainval.txt" \
    > "$OUT_ROOT/ImageSets/Main/train.txt"
cp "$V2007TE/ImageSets/Main/test.txt" "$OUT_ROOT/ImageSets/Main/voc2007test.txt"

n_train=$(wc -l < "$OUT_ROOT/ImageSets/Main/train.txt")
n_test=$(wc -l < "$OUT_ROOT/ImageSets/Main/voc2007test.txt")
n_images=$(find "$OUT_ROOT/JPEGImages" -name "*.jpg" | wc -l)

echo ""
echo "تم. $OUT_ROOT جاهز:"
echo "  train.txt (07trainval+12trainval): $n_train صورة (المتوقع 16,551)"
echo "  voc2007test.txt (07 test فقط):      $n_test صورة (المتوقع 4,952)"
echo "  إجمالي صور JPEGImages:              $n_images"
echo ""
echo "لا تنسَ: هذا تدريب من الصفر (بلا --resume) حتى تُعاد K-Means anchors"
echo "تلقائياً من توزيع صناديق 07+12 الجديد - سطر مطبوع في بداية train.py يؤكد ذلك."
