from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import OrientationDataset
from model import OrientationModel


ROOT = Path("data/mnt/ramdisk/max/90kDICT32px")
BATCH_SIZE = 256
VAL_LIMIT = 10_000

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def read_mjsynth_paths(annotation_file, limit=None):
    paths = []

    with open(annotation_file, "r", encoding="utf-8") as f:
        for line in f:
            relative_path = line.strip().split()[0]
            relative_path = relative_path.lstrip("./\\")

            path = ROOT / relative_path

            if path.exists():
                paths.append(path)

            if limit is not None and len(paths) >= limit:
                break

    return paths


val_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def main():

    # ----------------------
    # DATA
    # ----------------------

    val_paths = read_mjsynth_paths(
        ROOT / "annotation_val.txt",
        limit=VAL_LIMIT,
    )

    val_dataset = OrientationDataset(
        val_paths,
        transform=val_transform,
        train=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=DEVICE.type == "cuda",
    )

    # ----------------------
    # MODEL
    # ----------------------

    model = OrientationModel().to(DEVICE)

    checkpoint = torch.load(
        "best_mobilenetv3_orientation.pth",
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Loaded checkpoint:",
        "epoch =", checkpoint["epoch"],
        "best_brier =", checkpoint["best_brier"],
    )

    # ----------------------
    # VALIDATION
    # ----------------------

    normal_brier_sum = 0.0
    sym_brier_sum = 0.0

    normal_correct = 0
    sym_correct = 0

    total = 0

    use_bf16 = (
        DEVICE.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    with torch.no_grad():

        for images, labels in tqdm(val_loader):

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.float().to(
                DEVICE,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):

                # обычный прогноз
                logits_0 = model(images)

                # прогноз для rot180(x)
                images_180 = torch.rot90(
                    images,
                    k=2,
                    dims=(-2, -1),
                )

                logits_180 = model(images_180)

            # FP32 для расчёта вероятностей
            logits_0 = logits_0.float()
            logits_180 = logits_180.float()

            # ----------------------
            # обычный inference
            # ----------------------

            probs_normal = torch.sigmoid(
                logits_0
            )

            # ----------------------
            # rotation consistency
            # ----------------------

            sym_logits = (
                logits_0 - logits_180
            ) / 2.0

            probs_sym = torch.sigmoid(
                sym_logits
            )

            # ----------------------
            # METRICS
            # ----------------------

            normal_brier_sum += (
                (probs_normal - labels) ** 2
            ).sum().item()

            sym_brier_sum += (
                (probs_sym - labels) ** 2
            ).sum().item()

            normal_correct += (
                (probs_normal >= 0.5).float()
                == labels
            ).sum().item()

            sym_correct += (
                (probs_sym >= 0.5).float()
                == labels
            ).sum().item()

            total += labels.size(0)

    normal_brier = normal_brier_sum / total
    sym_brier = sym_brier_sum / total

    normal_acc = normal_correct / total
    sym_acc = sym_correct / total

    print()
    print("NORMAL")
    print(f"accuracy = {normal_acc:.6f}")
    print(f"Brier    = {normal_brier:.6f}")
    print(f"score    = {1 - normal_brier:.6f}")

    print()
    print("ROTATION CONSISTENCY")
    print(f"accuracy = {sym_acc:.6f}")
    print(f"Brier    = {sym_brier:.6f}")
    print(f"score    = {1 - sym_brier:.6f}")


if __name__ == "__main__":
    main()