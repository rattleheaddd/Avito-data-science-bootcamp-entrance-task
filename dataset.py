import random

from PIL import Image
from torch.utils.data import Dataset


# Preserve aspect ratio and pad to a fixed input size instead of stretching text,
# because geometric distortion may hurt orientation recognition.

def resize_with_padding(
    image,
    target_size=(192, 64),
    fill=(124, 116, 104),
):
    target_w, target_h = target_size

    w, h = image.size

    scale = min(
        target_w / w,
        target_h / h,
    )

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    image = image.resize(
        (new_w, new_h),
        Image.BILINEAR,
    )

    canvas = Image.new(
        "RGB",
        (target_w, target_h),
        fill,
    )

    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2

    canvas.paste(image, (x, y))

    return canvas


class OrientationDataset(Dataset):
    def __init__(self, image_paths, transform=None, train=True):
        self.image_paths = image_paths
        self.transform = transform
        self.train = train

    def __len__(self):
        if self.train:
            return len(self.image_paths)

        # validation:
        # каждый crop проверяем в 0° и 180°
        return len(self.image_paths) * 2

    def __getitem__(self, idx):

        if self.train:
            path = self.image_paths[idx]

            image = Image.open(path).convert("RGB")

            label = random.randint(0, 1)

        else:
            original_idx = idx // 2

            path = self.image_paths[original_idx]

            image = Image.open(path).convert("RGB")

            label = idx % 2

        if label == 1:
            image = image.rotate(180)

        image = resize_with_padding(
            image,
            target_size=(192, 64),
        )

        if self.transform:
            image = self.transform(image)

        return image, float(label)