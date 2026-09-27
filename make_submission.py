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

TEST_DIR = Path("data/test/images")
SAMPLE_SUBMISSION = Path("sample_submission.csv")

MJSYNTH_CHECKPOINT = Path(
    "best_mobilenetv3_orientation.pth"
)

TEXTOCR_CHECKPOINT = Path(
    "best_textocr_orientation.pth"
)

BATCH_SIZE = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# TRANSFORMS
# =========================================================

transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =========================================================
# DATASET
# =========================================================

class TestDataset(Dataset):

    def __init__(self, test_dir, image_ids):
        self.test_dir = Path(test_dir)
        self.image_ids = list(image_ids)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        path = self.test_dir / image_id

        if not path.exists():
            path = self.test_dir / f"{image_id}.png"

        image = Image.open(
            path
        ).convert("RGB")

        image = resize_with_padding(
            image,
            target_size=(192, 64),
        )

        image = transform(image)

        return image, image_id


# =========================================================
# MODEL
# =========================================================

def load_model(checkpoint_path):

    model = OrientationModel().to(DEVICE)

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
# ROTATION CONSISTENCY
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

        # z(x)
        logits_0 = model(images)

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

    # Приводим оба прогноза
    # к ориентации исходного x
    sym_logits = (
        logits_0.float()
        - logits_180.float()
    ) / 2.0

    probs = torch.sigmoid(
        sym_logits
    )

    return probs


# =========================================================
# MAIN
# =========================================================

def main():

    sample = pd.read_csv(
        SAMPLE_SUBMISSION
    )

    print(sample.head())
    print("Test images:", len(sample))

    # sample_submission задаёт нам
    # правильный порядок image_id
    image_ids = sample["image_id"].tolist()

    dataset = TestDataset(
        TEST_DIR,
        image_ids,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=True,
    )

    print("Loading MJSynth model...")

    model_mj = load_model(
        MJSYNTH_CHECKPOINT
    )

    print("Loading TextOCR model...")

    model_tx = load_model(
        TEXTOCR_CHECKPOINT
    )

    all_ids = []

    all_mj = []
    all_tx = []
    all_ensemble = []

    for images, ids in tqdm(
        loader,
        desc="Inference"
    ):

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        p_mj = predict(
            model_mj,
            images,
        )

        p_tx = predict(
            model_tx,
            images,
        )

        # 50 / 50 ensemble
        p_ensemble = (
            p_mj + p_tx
        ) / 2.0

        all_ids.extend(ids)

        all_mj.extend(
            p_mj.cpu().tolist()
        )

        all_tx.extend(
            p_tx.cpu().tolist()
        )

        all_ensemble.extend(
            p_ensemble.cpu().tolist()
        )

    # =====================================================
    # CREATE SUBMISSIONS
    # =====================================================

    submission_mj = pd.DataFrame({
        "image_id": all_ids,
        "p_180": all_mj,
    })

    submission_tx = pd.DataFrame({
        "image_id": all_ids,
        "p_180": all_tx,
    })

    submission_ensemble = pd.DataFrame({
        "image_id": all_ids,
        "p_180": all_ensemble,
    })

    # =====================================================
    # SANITY CHECKS
    # =====================================================

    for name, df in [
        ("MJSynth", submission_mj),
        ("TextOCR", submission_tx),
        ("Ensemble", submission_ensemble),
    ]:

        assert len(df) == len(sample)

        assert list(df.columns) == [
            "image_id",
            "p_180",
        ]

        assert not df["p_180"].isna().any()

        assert df["p_180"].between(
            0.0,
            1.0
        ).all()

        assert (
            df["image_id"].tolist()
            ==
            sample["image_id"].tolist()
        )

        print()
        print(name)
        print(
            "mean p180:",
            df["p_180"].mean()
        )

        print(
            "mean confidence:",
            (
                abs(
                    df["p_180"] - 0.5
                ) * 2
            ).mean()
        )

    # =====================================================
    # SAVE
    # =====================================================

    submission_mj.to_csv(
        "submission_mjsynth.csv",
        index=False,
    )

    submission_tx.to_csv(
        "submission_textocr.csv",
        index=False,
    )

    submission_ensemble.to_csv(
        "submission_ensemble.csv",
        index=False,
    )

    print()
    print("Saved:")
    print("submission_mjsynth.csv")
    print("submission_textocr.csv")
    print("submission_ensemble.csv")


if __name__ == "__main__":
    main()