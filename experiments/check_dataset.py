from pathlib import Path
from itertools import islice
import matplotlib.pyplot as plt

from dataset import OrientationDataset


root = Path("data/mnt/ramdisk/max/90kDICT32px")

print("Searching images...")

image_paths = list(
    islice(root.rglob("*.jpg"), 100)
)

print("Found:", len(image_paths))

if len(image_paths) == 0:
    raise RuntimeError("No jpg images found")

dataset = OrientationDataset(
    image_paths,
    transform=None,
    train=True,
)

print("Dataset created")

for i in range(10):
    image, label = dataset[i]

    print(i, label, image.size)

    plt.imshow(image)
    plt.title(f"label = {label}")
    plt.axis("off")
    plt.show()