import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path("data/textocr_dataset")

ANNOTATIONS = ROOT / "TextOCR_0.1_train.json"

# Проверь путь после распаковки.
# Внутри него должны существовать train/*.jpg
IMAGES_ROOT = ROOT / "train_images"


with open(ANNOTATIONS, "r", encoding="utf-8") as f:
    data = json.load(f)


imgs = data["imgs"]
anns = data["anns"]


# --------------------------
# собираем нормальные bbox
# --------------------------

valid_anns = []

for ann_id, ann in anns.items():

    text = ann["utf8_string"].strip()

    x, y, w, h = ann["bbox"]

    # Пока только самые очевидные плохие случаи
    if not text:
        continue

    if w < 5 or h < 5:
        continue

    valid_anns.append(ann_id)


print("All annotations:", len(anns))
print("Valid annotations:", len(valid_anns))


# --------------------------
# случайные 30 кропов
# --------------------------

random.seed(42)

sample = random.sample(
    valid_anns,
    30
)


fig, axes = plt.subplots(
    5,
    6,
    figsize=(16, 10)
)

axes = axes.flatten()


for ax, ann_id in zip(axes, sample):

    ann = anns[ann_id]

    image_id = ann["image_id"]

    image_info = imgs[image_id]

    image_path = (
        IMAGES_ROOT
        / Path(image_info["file_name"]).name
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    x, y, w, h = ann["bbox"]

    # PIL хочет left, upper, right, lower
    crop = image.crop((
        int(x),
        int(y),
        int(x + w),
        int(y + h),
    ))

    ax.imshow(crop)

    ax.set_title(
        repr(ann["utf8_string"]),
        fontsize=8
    )

    ax.axis("off")


plt.tight_layout()
plt.show()