from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

from model import OrientationModel
from dataset import resize_with_padding


SAMPLE_DIR = Path("test_random_100")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_BF16 = DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


MODELS = {
    "mjsynth": "best_mobilenetv3_orientation.pth",
    "textocr": "best_textocr_orientation.pth",
    "mjsynth_st": "best_mjsynth_synthtiger.pth",
    "textocr_st": "best_textocr_synthtiger.pth",
    "sym_v2": "best_mjsynth_synthtiger_sym_v2.pth",
}


def load_checkpoint(path):
    model = OrientationModel().to(DEVICE)

    checkpoint = torch.load(path, map_location=DEVICE)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value

    model.load_state_dict(cleaned)
    model.eval()

    return model


def prepare_image(path):
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

    x_rot = torch.rot90(
        x,
        k=2,
        dims=(-2, -1),
    )

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=USE_BF16,
    ):
        z = model(x)
        z_rot = model(x_rot)

    sym_logit = (
        z.float() - z_rot.float()
    ) / 2.0

    return torch.sigmoid(sym_logit).item()


def main():
    paths = sorted(
        p for p in SAMPLE_DIR.iterdir()
        if p.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".webp"
        }
    )

    print(f"Images: {len(paths)}")
    print(f"Device: {DEVICE}")

    models = {}

    for name, checkpoint in MODELS.items():
        print(f"Loading {name}: {checkpoint}")
        models[name] = load_checkpoint(checkpoint)

    rows = []

    for i, path in enumerate(paths, start=1):
        x = prepare_image(path)

        row = {
            "image_id": path.name,
        }

        for name, model in models.items():
            p = predict(model, x)

            row[f"p_{name}"] = p
            row[f"class_{name}"] = int(p >= 0.5)
            row[f"confidence_{name}"] = max(p, 1.0 - p)

        rows.append(row)

        if i % 20 == 0:
            print(f"{i}/{len(paths)}")

    df = pd.DataFrame(rows)

    final_name = "sym_v2"

    print("\n===== MODEL SUMMARY =====")

    for name in MODELS:
        p = df[f"p_{name}"]

        confidence = df[f"confidence_{name}"].mean()

        uncertain = (
            (p > 0.2) &
            (p < 0.8)
        ).mean()

        print(
            f"{name:12s} "
            f"mean_p={p.mean():.4f} "
            f"confidence={confidence:.4f} "
            f"uncertain={uncertain:.1%}"
        )

    print("\n===== AGREEMENT WITH FINAL sym_v2 =====")

    for name in MODELS:
        if name == final_name:
            continue

        agreement = (
            df[f"class_{name}"]
            == df[f"class_{final_name}"]
        ).mean()

        mean_diff = (
            df[f"p_{name}"]
            - df[f"p_{final_name}"]
        ).abs().mean()

        print(
            f"{name:12s} "
            f"agreement={agreement:.1%} "
            f"mean |Δp|={mean_diff:.4f}"
        )

    # Самые сильные расхождения final vs предыдущий лучший
    df["diff_final_vs_mjsynth_st"] = (
        df["p_sym_v2"] - df["p_mjsynth_st"]
    ).abs()

    print("\n===== BIGGEST CHANGES vs MJSynth -> SynthTIGER =====")

    biggest = df.sort_values(
        "diff_final_vs_mjsynth_st",
        ascending=False,
    ).head(20)

    print(
        biggest[
            [
                "image_id",
                "p_mjsynth_st",
                "p_sym_v2",
                "diff_final_vs_mjsynth_st",
            ]
        ].to_string(index=False)
    )

    df.to_csv(
        "test_model_comparison_v2.csv",
        index=False,
    )

    print("\nSaved: test_model_comparison_v2.csv")


if __name__ == "__main__":
    main()