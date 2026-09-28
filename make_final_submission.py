from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from model import OrientationModel
from dataset import resize_with_padding


TEST_DIR = Path("data/test/images")
SAMPLE_SUBMISSION = Path("sample_submission.csv")

CHECKPOINT = "weights/best_mjsynth_synthtiger_sym_v2.pth"
OUTPUT = "submission.csv"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_BF16 = (
    DEVICE.type == "cuda"
    and torch.cuda.is_bf16_supported()
)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


def load_model():
    model = OrientationModel(pretrained=False).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model" in checkpoint:
            checkpoint = checkpoint["model"]

    cleaned = {}

    for key, value in checkpoint.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value

    model.load_state_dict(cleaned)
    model.eval()

    return model


def prepare(path):
    image = Image.open(path).convert("RGB")

    image = resize_with_padding(
        image,
        target_size=(192, 64),
        fill=(124, 116, 104),
    )

    return transform(image)


@torch.no_grad()
def predict(model, x):
    x = x.unsqueeze(0).to(DEVICE)

    x180 = torch.rot90(
        x,
        k=2,
        dims=(-2, -1),
    )

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=USE_BF16,
    ):
        z0 = model(x)
        z180 = model(x180)

    # Final probability is computed from both orientations.
    # No temperature scaling is applied because calibration experiments
    # showed negligible improvement.
    sym_logit = (z0.float() - z180.float()) / 2.0
    p180 = torch.sigmoid(sym_logit)

    return p180.item()


def find_image(image_id):
    # В sample_submission ID были без .png
    candidates = [
        TEST_DIR / f"{image_id}.png",
        TEST_DIR / f"{image_id}.jpg",
        TEST_DIR / f"{image_id}.jpeg",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Image not found for {image_id}"
    )


def main():
    submission = pd.read_csv(SAMPLE_SUBMISSION)

    print(submission.head())
    print("Rows:", len(submission))

    model = load_model()

    predictions = []

    for i, image_id in enumerate(
        submission["image_id"],
        start=1,
    ):
        path = find_image(str(image_id))

        x = prepare(path)
        p = predict(model, x)

        predictions.append(p)

        if i % 1000 == 0:
            print(
                f"{i}/{len(submission)}"
            )

    submission["p_180"] = predictions

    submission.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print(f"Saved: {OUTPUT}")
    print(
        "mean p180:",
        submission["p_180"].mean(),
    )

    print(
        "mean confidence:",
        (
            submission["p_180"] - 0.5
        ).abs().mean() + 0.5,
    )

    print(
        "uncertain 0.2-0.8:",
        (
            (submission["p_180"] > 0.2)
            & (submission["p_180"] < 0.8)
        ).mean(),
    )

    print()
    print(submission.head())


if __name__ == "__main__":
    main()