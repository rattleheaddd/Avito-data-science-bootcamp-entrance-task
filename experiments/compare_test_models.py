from pathlib import Path

import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import OrientationModel
from dataset import resize_with_padding


# =========================================================
# CONFIG
# =========================================================

TEST_DIR = Path("test_random_100")

MJSYNTH_CHECKPOINT = (
    "best_mobilenetv3_orientation.pth"
)

TEXTOCR_CHECKPOINT = (
    "best_textocr_orientation.pth"
)

BATCH_SIZE = 100

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# DATASET
# =========================================================

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class TestDataset(Dataset):

    def __init__(self, root):
        self.paths = sorted([
            p for p in root.iterdir()
            if p.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
                ".bmp",
            }
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        path = self.paths[idx]

        image = Image.open(
            path
        ).convert("RGB")

        image = resize_with_padding(
            image,
            target_size=(192, 64),
        )

        image = transform(image)

        return image, path.name


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(checkpoint_path):

    model = OrientationModel().to(
        DEVICE
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# =========================================================
# INFERENCE
# =========================================================

@torch.no_grad()
def predict(model, images):

    use_bf16 = (
        DEVICE.type == "cuda"
        and torch.cuda.is_bf16_supported()
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

    # rotation consistency
    sym_logits = (
        logits_0.float()
        - logits_180.float()
    ) / 2.0

    # T = 1
    probs = torch.sigmoid(
        sym_logits
    )

    return probs


# =========================================================
# MAIN
# =========================================================

def main():

    dataset = TestDataset(TEST_DIR)

    print(
        "Test images:",
        len(dataset)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Loading MJSynth model...")

    mjsynth_model = load_model(
        MJSYNTH_CHECKPOINT
    )

    print("Loading TextOCR model...")

    textocr_model = load_model(
        TEXTOCR_CHECKPOINT
    )

    rows = []

    for images, filenames in tqdm(loader):

        images = images.to(DEVICE)

        p_mjsynth = predict(
            mjsynth_model,
            images,
        ).cpu()

        p_textocr = predict(
            textocr_model,
            images,
        ).cpu()

        for name, p1, p2 in zip(
            filenames,
            p_mjsynth,
            p_textocr,
        ):

            p1 = p1.item()
            p2 = p2.item()

            class1 = int(p1 >= 0.5)
            class2 = int(p2 >= 0.5)

            rows.append({
                "image_id": name,

                "p180_mjsynth":
                    p1,

                "p180_textocr":
                    p2,

                "class_mjsynth":
                    class1,

                "class_textocr":
                    class2,

                "agree":
                    int(class1 == class2),

                "prob_diff":
                    abs(p1 - p2),

                # 0 = полностью не уверена
                # 1 = максимально уверена
                "confidence_mjsynth":
                    abs(p1 - 0.5) * 2,

                "confidence_textocr":
                    abs(p2 - 0.5) * 2,
            })

    df = pd.DataFrame(rows)

    df = df.sort_values(
        "prob_diff",
        ascending=False,
    )

    df.to_csv(
        "test_model_comparison.csv",
        index=False,
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    agreement = df["agree"].mean()

    mean_conf_mj = (
        df["confidence_mjsynth"].mean()
    )

    mean_conf_tx = (
        df["confidence_textocr"].mean()
    )

    uncertain_mj = (
        (
            (df["p180_mjsynth"] > 0.2)
            &
            (df["p180_mjsynth"] < 0.8)
        )
        .mean()
    )

    uncertain_tx = (
        (
            (df["p180_textocr"] > 0.2)
            &
            (df["p180_textocr"] < 0.8)
        )
        .mean()
    )

    strong_disagreement = (
        df["prob_diff"] > 0.5
    ).sum()

    print()
    print("=" * 60)

    print(
        f"Class agreement: "
        f"{agreement:.1%}"
    )

    print()

    print("MEAN CONFIDENCE")

    print(
        f"MJSynth : "
        f"{mean_conf_mj:.3f}"
    )

    print(
        f"TextOCR : "
        f"{mean_conf_tx:.3f}"
    )

    print()

    print("UNCERTAIN (0.2 < p < 0.8)")

    print(
        f"MJSynth : "
        f"{uncertain_mj:.1%}"
    )

    print(
        f"TextOCR : "
        f"{uncertain_tx:.1%}"
    )

    print()

    print(
        "Strong probability disagreements "
        f"(|p1-p2| > 0.5): "
        f"{strong_disagreement}"
    )

    print()

    print("TOP DISAGREEMENTS")

    print(
        df[
            [
                "image_id",
                "p180_mjsynth",
                "p180_textocr",
                "prob_diff",
                "agree",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print()
    print(
        "Saved test_model_comparison.csv"
    )


if __name__ == "__main__":
    main()