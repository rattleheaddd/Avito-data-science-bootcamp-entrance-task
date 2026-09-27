from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import OrientationModel
from textocr_dataset import TextOCRDataset


# =========================================================
# CONFIG
# =========================================================

ROOT = Path("data/textocr_dataset")

ANNOTATIONS = (
    ROOT / "TextOCR_0.1_val.json"
)

IMAGES_ROOT = (
    ROOT / "train_images"
)

CHECKPOINT = "best_textocr_orientation.pth"

BATCH_SIZE = 256

LIMIT = None


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# TRANSFORM
# =========================================================

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

    # -----------------------------
    # DATASET
    # -----------------------------

    dataset = TextOCRDataset(
        annotation_path=ANNOTATIONS,
        images_root=IMAGES_ROOT,
        transform=val_transform,
        train=False,
        limit=LIMIT,
    )

    print(
        "Validation examples:",
        len(dataset)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=True,
    )

    # -----------------------------
    # MODEL
    # -----------------------------

    model = OrientationModel().to(
        DEVICE
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Loaded checkpoint:",
        "epoch =",
        checkpoint["epoch"],
        "best_brier =",
        checkpoint["best_brier"],
    )

    # -----------------------------
    # BF16
    # -----------------------------

    use_bf16 = (
        DEVICE.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    # -----------------------------
    # METRICS
    # -----------------------------

    normal_brier_sum = 0.0
    sym_brier_sum = 0.0

    normal_correct = 0
    sym_correct = 0

    total = 0

    # -----------------------------
    # VALIDATION
    # -----------------------------

    with torch.no_grad():

        progress = tqdm(loader)

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

            # дальше FP32
            logits_0 = logits_0.float()
            logits_180 = (
                logits_180.float()
            )

            # =================================================
            # NORMAL
            # =================================================

            probs_normal = torch.sigmoid(
                logits_0
            )

            # =================================================
            # ROTATION CONSISTENCY
            # =================================================

            sym_logits = (
                logits_0
                - logits_180
            ) / 2.0

            probs_sym = torch.sigmoid(
                sym_logits
            )

            # =================================================
            # BRIER
            # =================================================

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

            # =================================================
            # ACC
            # =================================================

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

    # =========================================================
    # RESULTS
    # =========================================================

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
    print("=" * 50)

    print("NORMAL")
    print(
        f"accuracy = {normal_acc:.6f}"
    )
    print(
        f"Brier    = {normal_brier:.6f}"
    )
    print(
        f"score    = "
        f"{1 - normal_brier:.6f}"
    )

    print()

    print("ROTATION CONSISTENCY")
    print(
        f"accuracy = {sym_acc:.6f}"
    )
    print(
        f"Brier    = {sym_brier:.6f}"
    )
    print(
        f"score    = "
        f"{1 - sym_brier:.6f}"
    )


if __name__ == "__main__":
    main()