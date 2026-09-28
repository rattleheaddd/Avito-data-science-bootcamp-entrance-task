from pathlib import Path
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataset import resize_with_padding


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Very short/non-alphanumeric strings often contain too little
# orientation information, so we discard them.
def valid_text(text: str) -> bool:
    return sum(c.isalnum() for c in text) >= 2


def load_synthtiger_samples(root):
    root = Path(root)
    gt_path = root / "gt.txt"

    samples = []

    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) != 2:
                continue

            rel_path, text = parts

            if not valid_text(text):
                continue

            image_path = root / Path(rel_path)

            if image_path.exists():
                samples.append((image_path, text))

    return samples


def split_samples(samples, val_fraction=0.05, seed=42):
    """
    Deterministically split SynthTIGER samples into train and validation subsets.
    The fixed seed ensures reproducible model comparisons.
    """
    samples = list(samples)

    rng = random.Random(seed)
    rng.shuffle(samples)

    n_val = int(len(samples) * val_fraction)

    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    return train_samples, val_samples


class SynthTigerDataset(Dataset):
    def __init__(
        self,
        samples,
        train=True,
        target_size=(192, 64),
    ):
        self.samples = samples
        self.train = train
        self.target_size = target_size

        if train:
            self.transform = transforms.Compose([
                transforms.RandomAffine(
                    degrees=3,
                    translate=(0.02, 0.02),
                    scale=(0.95, 1.05),
                ),
                transforms.ColorJitter(
                    brightness=0.10,
                    contrast=0.10,
                    saturation=0.08,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                ),
            ])

    def __len__(self):
        if self.train:
            return len(self.samples)

        # val: каждая исходная картинка два раза
        # upright + rot180
        return len(self.samples) * 2

    def __getitem__(self, idx):
        if self.train:
            sample_idx = idx

            # SynthTIGER images are generated upright by construction.
            # We create clean orientation labels on the fly:
            # 0 = original orientation, 1 = rotated by 180 degrees.
            label = random.randint(0, 1)
        else:
            # Each validation image is evaluated in both orientations.
            # This gives a balanced and deterministic validation set.
            sample_idx = idx // 2

            # 0 -> upright
            # 1 -> rot180
            label = idx % 2

        image_path, _ = self.samples[sample_idx]

        image = Image.open(image_path).convert("RGB")

        if label == 1:
            image = image.transpose(Image.Transpose.ROTATE_180)
        
        # Preserve aspect ratio and pad to a fixed input size instead of stretching text,
        # because geometric distortion may hurt orientation recognition.

        image = resize_with_padding(
            image,
            target_size=self.target_size,
            fill=(124, 116, 104),
        )

        image = self.transform(image)

        return image, float(label)