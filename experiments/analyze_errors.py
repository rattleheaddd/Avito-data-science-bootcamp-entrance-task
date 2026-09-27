from pathlib import Path

import matplotlib.pyplot as plt
import torch

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import OrientationModel
from textocr_dataset import TextOCRDataset


ROOT = Path("data/textocr_dataset")

ANNOTATIONS = ROOT / "TextOCR_0.1_val.json"
IMAGES_ROOT = ROOT / "train_images"

CHECKPOINT = "best_textocr_orientation.pth"

BATCH_SIZE = 256
TOP_K = 50

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


val_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def get_raw_crop(dataset, sample_idx):

    filename, bbox, text = dataset.samples[sample_idx]

    image_path = (
        dataset.images_root
        / filename
    )

    image = Image.open(
        image_path
    ).convert("RGB")

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

    crop = image.crop(
        (left, top, right, bottom)
    )

    return crop, text


def main():

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

    # --------------------------
    # MODEL
    # --------------------------

    model = OrientationModel().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    use_bf16 = (
        DEVICE.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    # в errors
    # (brier_error, probability, sample_idx)
    errors = []

    global_idx = 0

    # --------------------------
    # INFERENCE
    # --------------------------

    with torch.no_grad():

        for images, labels in tqdm(
            loader,
            desc="Analyzing errors"
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

                logits_0 = model(images)

                images_180 = torch.rot90(
                    images,
                    k=2,
                    dims=(-2, -1),
                )

                logits_180 = model(
                    images_180
                )

            sym_logits = (
                logits_0.float()
                - logits_180.float()
            ) / 2.0

            probs = torch.sigmoid(
                sym_logits
            ).cpu()

            batch_size = len(probs)

            for j in range(batch_size):

                dataset_idx = (
                    global_idx + j
                )

                # Только оригинальный crop:
                # even index -> label 0
                #
                # odd index -> тот же crop,
                # повернутый на 180°.
                if dataset_idx % 2 != 0:
                    continue

                p180 = probs[j].item()

                # y = 0,
                # поэтому Brier = p^2
                brier_error = p180 ** 2

                sample_idx = (
                    dataset_idx // 2
                )

                errors.append(
                    (
                        brier_error,
                        p180,
                        sample_idx,
                    )
                )

            global_idx += batch_size

    # --------------------------
    # ONLY WRONG PREDICTIONS
    # --------------------------

    wrong = [
        x
        for x in errors
        if x[1] >= 0.5
    ]

    wrong.sort(
        key=lambda x: x[0],
        reverse=True
    )

    print()
    print(
        "Original crops:",
        len(errors)
    )

    print(
        "Wrong original crops:",
        len(wrong)
    )

    print(
        "Error rate:",
        len(wrong) / len(errors)
    )

    worst = wrong[:TOP_K]

    # --------------------------
    # SHOW GRID
    # --------------------------

    cols = 5

    rows = (
        len(worst) + cols - 1
    ) // cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(15, rows * 2.5),
    )

    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for ax, (
        brier,
        probability,
        sample_idx,
    ) in zip(axes, worst):

        crop, text = get_raw_crop(
            dataset,
            sample_idx,
        )

        ax.imshow(crop)

        ax.set_title(
            f"{text!r}\n"
            f"p180={probability:.3f} "
            f"Brier={brier:.3f}",
            fontsize=8,
        )

        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        "textocr_worst_errors.png",
        dpi=160,
    )

    plt.show()


if __name__ == "__main__":
    main()