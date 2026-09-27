from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import OrientationModel
from textocr_dataset import TextOCRDataset


# =========================================================
# CONFIG
# =========================================================

ROOT = Path("data/textocr_dataset")

TRAIN_ANNOTATIONS = (
    ROOT / "TextOCR_0.1_train.json"
)

VAL_ANNOTATIONS = (
    ROOT / "TextOCR_0.1_val.json"
)

IMAGES_ROOT = (
    ROOT / "train_images"
)

PRETRAINED_CHECKPOINT = "best_textocr_orientation.pth"

OUTPUT_CHECKPOINT = (
    "best_textocr_orientation_final.pth"
)

LAST_CHECKPOINT = "last_textocr_checkpoint.pth"


BATCH_SIZE = 256

TRAIN_LIMIT = 800_000
EPOCHS = 2
LR = 2e-5

# Потом прогнать полный val
VAL_LIMIT = None


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# TRANSFORMS
# =========================================================

train_transform = transforms.Compose([

    transforms.RandomAffine(
        degrees=8,
        translate=(0.04, 0.04),
        scale=(0.9, 1.1),
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
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


# =========================================================
# MAIN
# =========================================================

def main():

    print("Device:", DEVICE)

    # =====================================================
    # DATA
    # =====================================================

    train_dataset = TextOCRDataset(
        annotation_path=TRAIN_ANNOTATIONS,
        images_root=IMAGES_ROOT,
        transform=train_transform,
        train=True,
        limit=TRAIN_LIMIT,
        seed=42,
    )

    val_dataset = TextOCRDataset(
        annotation_path=VAL_ANNOTATIONS,
        images_root=IMAGES_ROOT,
        transform=val_transform,
        train=False,
        limit=VAL_LIMIT,
        seed=123,
    )

    print(
        "Train examples:",
        len(train_dataset)
    )

    print(
        "Val examples:",
        len(val_dataset)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,

        # На Windows с огромным JSON/Dataset
        # лучше сначала не ставить слишком много.
        num_workers=4,

        pin_memory=DEVICE.type == "cuda",
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=True,
    )

    # =====================================================
    # MODEL
    # =====================================================

    model = OrientationModel().to(
        DEVICE
    )

    # -----------------------------------------
    # Загружаем веса после MJSynth
    # -----------------------------------------

    checkpoint = torch.load(
        PRETRAINED_CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Loaded MJSynth checkpoint:",
        "epoch =",
        checkpoint["epoch"],
        "Brier =",
        checkpoint["best_brier"],
    )

    # =====================================================
    # OPTIMIZER
    # =====================================================

    criterion = nn.BCEWithLogitsLoss()

    # optimizer создаём НОВЫЙ
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    # =====================================================
    # BF16
    # =====================================================

    use_bf16 = (
        DEVICE.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    print(
        "BF16:",
        use_bf16
    )

    # Мы используем rotation consistency
    # как финальный inference,
    # поэтому сохраняем модель именно по нему.
    best_brier = float("inf")

    # =====================================================
    # EPOCHS
    # =====================================================

    for epoch in range(EPOCHS):

        # =================================================
        # TRAIN
        # =================================================

        model.train()

        train_loss_sum = 0.0
        train_count = 0

        progress = tqdm(
            train_loader,
            desc=(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"[train]"
            ),
        )

        for images, labels in progress:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.float().to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):

                logits = model(images)

                loss = criterion(
                    logits,
                    labels,
                )

            loss.backward()

            optimizer.step()

            batch_size = images.size(0)

            train_loss_sum += (
                loss.item()
                * batch_size
            )

            train_count += batch_size

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = (
            train_loss_sum
            / train_count
        )

        # =================================================
        # VALIDATION
        # =================================================

        model.eval()

        normal_brier_sum = 0.0
        sym_brier_sum = 0.0

        normal_correct = 0
        sym_correct = 0

        total = 0

        with torch.no_grad():

            progress = tqdm(
                val_loader,
                desc=(
                    f"Epoch {epoch + 1}/{EPOCHS} "
                    f"[val]"
                ),
            )

            for images, labels in progress:

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

                    # x
                    logits_0 = model(
                        images
                    )

                    # rot180(x)
                    images_180 = torch.rot90(
                        images,
                        k=2,
                        dims=(-2, -1),
                    )

                    logits_180 = model(
                        images_180
                    )

                logits_0 = (
                    logits_0.float()
                )

                logits_180 = (
                    logits_180.float()
                )

                # --------------------------
                # NORMAL
                # --------------------------

                probs_normal = (
                    torch.sigmoid(
                        logits_0
                    )
                )

                # --------------------------
                # ROTATION CONSISTENCY
                # --------------------------

                sym_logits = (
                    logits_0
                    - logits_180
                ) / 2.0

                probs_sym = torch.sigmoid(
                    sym_logits
                )

                # --------------------------
                # METRICS
                # --------------------------

                normal_brier_sum += (
                    (
                        probs_normal
                        - labels
                    ) ** 2
                ).sum().item()

                sym_brier_sum += (
                    (
                        probs_sym
                        - labels
                    ) ** 2
                ).sum().item()

                normal_correct += (
                    (
                        probs_normal >= 0.5
                    ).float()
                    == labels
                ).sum().item()

                sym_correct += (
                    (
                        probs_sym >= 0.5
                    ).float()
                    == labels
                ).sum().item()

                total += labels.size(0)

        # =================================================
        # RESULTS
        # =================================================

        normal_brier = (
            normal_brier_sum / total
        )

        sym_brier = (
            sym_brier_sum / total
        )

        normal_acc = (
            normal_correct / total
        )

        sym_acc = (
            sym_correct / total
        )

        print()
        print("=" * 60)

        print(
            f"Epoch {epoch + 1}"
        )

        print(
            f"train_loss = "
            f"{train_loss:.6f}"
        )

        print()

        print("NORMAL")
        print(
            f"accuracy = "
            f"{normal_acc:.6f}"
        )
        print(
            f"Brier    = "
            f"{normal_brier:.6f}"
        )
        print(
            f"score    = "
            f"{1 - normal_brier:.6f}"
        )

        print()

        print(
            "ROTATION CONSISTENCY"
        )

        print(
            f"accuracy = "
            f"{sym_acc:.6f}"
        )

        print(
            f"Brier    = "
            f"{sym_brier:.6f}"
        )

        print(
            f"score    = "
            f"{1 - sym_brier:.6f}"
        )

        # =================================================
        # SAVE BEST
        # =================================================
        if sym_brier < best_brier:

            best_brier = sym_brier

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch + 1,

                    "best_brier":
                        best_brier,

                    "source":
                        "TextOCR fine-tune",
                },
                OUTPUT_CHECKPOINT,
            )

            print()
            print(
                "Saved best TextOCR model "
                f"(Brier={best_brier:.6f})"
            )
        print("=" * 60)
        print()
        
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),

                "epoch": epoch + 1,

                "best_brier": best_brier,

                "source": "TextOCR fine-tune",
            },
            LAST_CHECKPOINT,
        )

        print(
            f"Saved last checkpoint "
            f"(epoch={epoch + 1})"
        )


if __name__ == "__main__":
    main()