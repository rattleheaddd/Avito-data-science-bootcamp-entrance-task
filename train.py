from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import OrientationDataset
from model import OrientationModel


ROOT = Path("data/mnt/ramdisk/max/90kDICT32px")

BATCH_SIZE = 256
EPOCHS = 5
LR = 3e-4

# Для первого запуска не берём миллионы
TRAIN_LIMIT = 200_000
VAL_LIMIT = 10_000


def read_mjsynth_paths(annotation_file, limit=None):
    paths = []

    with open(annotation_file, "r", encoding="utf-8") as f:
        for line in f:
            # Нам нужен только первый элемент строки — путь к картинке
            relative_path = line.strip().split()[0]

            # Обычно MJSynth пишет пути с ./ в начале
            relative_path = relative_path.lstrip("./\\")

            path = ROOT / relative_path

            if path.exists():
                paths.append(path)

            if limit is not None and len(paths) >= limit:
                break

    return paths


train_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=10,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1),
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


val_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # -------------------------
    # PATHS
    # -------------------------

    train_paths = read_mjsynth_paths(
        ROOT / "annotation_train.txt",
        limit=TRAIN_LIMIT,
    )

    val_paths = read_mjsynth_paths(
        ROOT / "annotation_val.txt",
        limit=VAL_LIMIT,
    )

    print("Train images:", len(train_paths))
    print("Val original images:", len(val_paths))

    # -------------------------
    # DATASETS
    # -------------------------

    train_dataset = OrientationDataset(
        train_paths,
        transform=train_transform,
        train=True,
    )

    val_dataset = OrientationDataset(
        val_paths,
        transform=val_transform,
        train=False,
    )

    print("Train examples:", len(train_dataset))
    print("Val examples:", len(val_dataset))

    # -------------------------
    # DATALOADERS
    # -------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    # -------------------------
    # MODEL
    # -------------------------

    model = OrientationModel().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    # Mixed precision
    use_bf16 = (
    device.type == "cuda"
    and torch.cuda.is_bf16_supported())


    # =========================
    # TRAINING
    # =========================
    best_brier = float("inf")
    for epoch in range(EPOCHS):

        # ---------------------
        # TRAIN
        # ---------------------

        model.train()

        train_loss = 0.0
        train_count = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS} [train]",
        )

        for images, labels in progress:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.float().to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)

            train_loss += (
                loss.item() * batch_size
            )

            train_count += batch_size

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss /= train_count

        # ---------------------
        # VALIDATION
        # ---------------------

        model.eval()

        val_loss = 0.0
        val_count = 0

        brier_sum = 0.0
        correct = 0

        with torch.no_grad():

            progress = tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{EPOCHS} [val]",
            )

            for images, labels in progress:

                images = images.to(
                    device,
                    non_blocking=True,
                )

                labels = labels.float().to(
                    device,
                    non_blocking=True,
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=use_bf16,
                ):
                    logits = model(images)
                    loss = criterion(logits, labels)

                probs = torch.sigmoid(logits)

                predictions = (
                    probs >= 0.5
                ).float()

                batch_size = images.size(0)

                val_loss += (
                    loss.item() * batch_size
                )

                brier_sum += (
                    (probs - labels) ** 2
                ).sum().item()

                correct += (
                    predictions == labels
                ).sum().item()

                val_count += batch_size

        val_loss /= val_count

        brier = brier_sum / val_count
        accuracy = correct / val_count

        score = 1.0 - brier

        print()
        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"acc={accuracy:.4f} | "
            f"Brier={brier:.6f} | "
            f"score={score:.6f}"
        )

        # ---------------------
        # SAVE BEST (and last)
        # ---------------------

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_brier": best_brier,
            },
        "last_checkpoint.pth"
        )

        if brier < best_brier:

            best_brier = brier

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "best_brier":
                        best_brier,

                    "epoch":
                        epoch + 1,
                },
                "best_mobilenetv3_orientation.pth",
            )

            print(
                f"Saved best model "
                f"(Brier={best_brier:.6f})"
            )

        print()


if __name__ == "__main__":
    main()