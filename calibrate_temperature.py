import json
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

ANNOTATIONS = ROOT / "TextOCR_0.1_val.json"
IMAGES_ROOT = ROOT / "train_images"

CHECKPOINT = "best_textocr_orientation.pth"

BATCH_SIZE = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# TRANSFORMS
# =========================================================

val_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =========================================================
# TEMPERATURE SEARCH
# =========================================================

def brier_score(logits, labels, T):
    probs = torch.sigmoid(
        logits / T
    )

    return torch.mean(
        (probs - labels) ** 2
    ).item()


def find_best_temperature(logits, labels):

    logits = logits.to(DEVICE)
    labels = labels.to(DEVICE)

    # -------------------------
    # грубый поиск
    # -------------------------

    best_T = 1.0
    best_brier = float("inf")

    temperatures = torch.linspace(
        0.25,
        4.0,
        151,
        device=DEVICE,
    )

    for T in temperatures:

        brier = brier_score(
            logits,
            labels,
            T,
        )

        if brier < best_brier:
            best_brier = brier
            best_T = T.item()

    # -------------------------
    # точный поиск вокруг best
    # -------------------------

    left = max(
        0.05,
        best_T - 0.2,
    )

    right = best_T + 0.2

    temperatures = torch.linspace(
        left,
        right,
        401,
        device=DEVICE,
    )

    for T in temperatures:

        brier = brier_score(
            logits,
            labels,
            T,
        )

        if brier < best_brier:
            best_brier = brier
            best_T = T.item()

    return best_T, best_brier


# =========================================================
# MAIN
# =========================================================

def main():

    print("Device:", DEVICE)

    # =====================================================
    # DATASET
    # =====================================================

    dataset = TextOCRDataset(
        annotation_path=ANNOTATIONS,
        images_root=IMAGES_ROOT,
        transform=val_transform,
        train=False,
        limit=None,
        seed=123,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=True,
    )

    print(
        "Validation examples:",
        len(dataset)
    )

    # =====================================================
    # MODEL
    # =====================================================

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
        "Loaded model:",
        CHECKPOINT
    )

    # =====================================================
    # COLLECT SYMMETRIC LOGITS
    # =====================================================

    use_bf16 = (
        DEVICE.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    all_logits = []
    all_labels = []

    with torch.no_grad():

        for images, labels in tqdm(
            loader,
            desc="Collecting logits"
        ):

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):

                # z(x)
                logits_0 = model(
                    images
                )

                # rot180(x)
                images_180 = torch.rot90(
                    images,
                    k=2,
                    dims=(-2, -1),
                )

                # z(rot180(x))
                logits_180 = model(
                    images_180
                )

            # rotation consistency
            sym_logits = (
                logits_0.float()
                - logits_180.float()
            ) / 2.0

            all_logits.append(
                sym_logits.cpu()
            )

            all_labels.append(
                labels.float().cpu()
            )

    all_logits = torch.cat(
        all_logits
    )

    all_labels = torch.cat(
        all_labels
    )

    print(
        "Collected:",
        len(all_logits)
    )

    # =====================================================
    # SPLIT BY ORIGINAL CROP
    # =====================================================

    # Dataset устроен так:
    #
    # crop0 normal
    # crop0 rotated
    # crop1 normal
    # crop1 rotated
    # ...

    assert len(all_logits) % 2 == 0

    pair_logits = all_logits.reshape(
        -1,
        2
    )

    pair_labels = all_labels.reshape(
        -1,
        2
    )

    num_pairs = len(pair_logits)

    generator = torch.Generator()
    generator.manual_seed(42)

    permutation = torch.randperm(
        num_pairs,
        generator=generator,
    )

    split = num_pairs // 2

    calibration_idx = permutation[:split]
    holdout_idx = permutation[split:]

    calibration_logits = (
        pair_logits[calibration_idx]
        .reshape(-1)
    )

    calibration_labels = (
        pair_labels[calibration_idx]
        .reshape(-1)
    )

    holdout_logits = (
        pair_logits[holdout_idx]
        .reshape(-1)
    )

    holdout_labels = (
        pair_labels[holdout_idx]
        .reshape(-1)
    )

    print(
        "Calibration examples:",
        len(calibration_logits)
    )

    print(
        "Holdout examples:",
        len(holdout_logits)
    )

    # =====================================================
    # BEFORE CALIBRATION
    # =====================================================

    calibration_before = brier_score(
        calibration_logits,
        calibration_labels,
        1.0,
    )

    holdout_before = brier_score(
        holdout_logits,
        holdout_labels,
        1.0,
    )

    # =====================================================
    # FIND T
    # =====================================================

    best_T, calibration_after = (
        find_best_temperature(
            calibration_logits,
            calibration_labels,
        )
    )

    # =====================================================
    # HOLDOUT TEST
    # =====================================================

    holdout_after = brier_score(
        holdout_logits,
        holdout_labels,
        best_T,
    )

    print()
    print("=" * 60)

    print(
        f"Best T = {best_T:.4f}"
    )

    print()

    print("CALIBRATION HALF")

    print(
        f"Brier before = "
        f"{calibration_before:.6f}"
    )

    print(
        f"Brier after  = "
        f"{calibration_after:.6f}"
    )

    print()

    print("HOLDOUT HALF")

    print(
        f"Brier before = "
        f"{holdout_before:.6f}"
    )

    print(
        f"Brier after  = "
        f"{holdout_after:.6f}"
    )

    print("=" * 60)

    # =====================================================
    # FINAL T ON FULL VALIDATION
    # =====================================================

    # Если calibration реально помогает на holdout,
    # можем использовать весь val для финальной оценки T.

    if holdout_after < holdout_before:

        print()
        print(
            "Calibration improves holdout."
        )

        print(
            "Finding final T on full validation..."
        )

        final_T, full_brier = (
            find_best_temperature(
                all_logits,
                all_labels,
            )
        )

    else:

        print()
        print(
            "Calibration did NOT improve holdout."
        )

        print(
            "Using T = 1.0"
        )

        final_T = 1.0

        full_brier = brier_score(
            all_logits,
            all_labels,
            1.0,
        )

    print()
    print(
        f"FINAL T = {final_T:.4f}"
    )

    print(
        f"Full-val Brier = "
        f"{full_brier:.6f}"
    )

    # =====================================================
    # SAVE T
    # =====================================================

    result = {
        "temperature": final_T,
        "calibration_temperature": best_T,

        "holdout_brier_before":
            holdout_before,

        "holdout_brier_after":
            holdout_after,

        "full_val_brier":
            full_brier,
    }

    with open(
        "temperature.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
        )

    print()
    print(
        "Saved temperature.json"
    )


if __name__ == "__main__":
    main()