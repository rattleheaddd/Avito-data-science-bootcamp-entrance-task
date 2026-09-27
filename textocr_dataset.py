import json
import random
from functools import lru_cache
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from dataset import resize_with_padding


class TextOCRDataset(Dataset):
    def __init__(
        self,
        annotation_path,
        images_root,
        transform=None,
        train=False,
        limit=None,
        seed=42,
    ):
        self.images_root = Path(images_root)
        self.transform = transform
        self.train = train

        print("Loading TextOCR annotations...")

        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        imgs = data["imgs"]
        anns = data["anns"]

        samples = []

        for ann in anns.values():

            text = ann["utf8_string"].strip()
            x, y, w, h = ann["bbox"]

            # слишком маленький bbox
            if w < 8 or h < 8:
                continue

            # хотя бы 2 буквы/цифры
            alnum_count = sum(
                c.isalnum()
                for c in text
            )

            if alnum_count < 2:
                continue

            image_id = ann["image_id"]

            filename = Path(
                imgs[image_id]["file_name"]
            ).name

            samples.append(
                (
                    filename,
                    (x, y, w, h),
                    text,
                )
            )

        # ВАЖНО:
        # перемешиваем ДО limit,
        # иначе первые 300k будут неслучайными
        rng = random.Random(seed)
        rng.shuffle(samples)

        if limit is not None:
            samples = samples[:limit]

        self.samples = samples

        print(
            f"Valid samples: {len(self.samples)}"
        )

        # Большой JSON больше не нужен
        del data
        del imgs
        del anns

    def __len__(self):

        if self.train:
            return len(self.samples)

        return 2 * len(self.samples)

    @lru_cache(maxsize=32)
    def _load_image(self, image_path):

        return Image.open(
            image_path
        ).convert("RGB")

    def __getitem__(self, idx):

        if self.train:

            sample_idx = idx

            # каждый epoch ориентация может меняться
            label = random.randint(0, 1)

        else:

            sample_idx = idx // 2
            label = idx % 2

        filename, bbox, text = self.samples[sample_idx]

        image_path = (
            self.images_root
            / filename
        )

        image = self._load_image(
            str(image_path)
        )

        x, y, w, h = bbox

        left = max(0, int(x))
        top = max(0, int(y))

        right = min(
            image.width,
            int(x + w)
        )

        bottom = min(
            image.height,
            int(y + h)
        )

        crop = image.crop((
            left,
            top,
            right,
            bottom,
        ))

        if label == 1:
            crop = crop.rotate(180)

        crop = resize_with_padding(
            crop,
            target_size=(192, 64),
        )

        if self.transform:
            crop = self.transform(crop)

        return crop, float(label)