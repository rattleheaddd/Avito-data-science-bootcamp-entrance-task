from pathlib import Path
from sklearn.model_selection import train_test_split


root = Path("data/mnt/ramdisk/max/90kDICT32px")

image_paths = (
    list(root.rglob("*.jpg"))
    + list(root.rglob("*.png"))
    + list(root.rglob("*.jpeg"))
)

print(f"Total images: {len(image_paths)}")


# Пока baseline
image_paths = image_paths[:200_000]


train_paths, val_paths = train_test_split(
    image_paths,
    test_size=0.05,
    random_state=42,
)

print("Train:", len(train_paths))
print("Val:", len(val_paths))