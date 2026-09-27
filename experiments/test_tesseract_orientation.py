import random
from pathlib import Path

import pandas as pd
import pytesseract

from PIL import Image
from rapidfuzz.fuzz import ratio

from textocr_dataset import TextOCRDataset


ROOT = Path("data/textocr_dataset")

ANNOTATIONS = ROOT / "TextOCR_0.1_val.json"
IMAGES_ROOT = ROOT / "train_images"

N = 300
SEED = 42


# Если tesseract.exe не добавлен в PATH:
#
# pytesseract.pytesseract.tesseract_cmd = (
#     r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# )


def normalize_text(s):
    """
    Оставляем только буквы и цифры,
    приводим к lowercase.
    """
    return "".join(
        c.lower()
        for c in s
        if c.isalnum()
    )


def run_tesseract(image):
    data = pytesseract.image_to_data(
        image,
        config="--psm 7",
        output_type=pytesseract.Output.DICT,
    )

    words = []
    confs = []

    for text, conf in zip(
        data["text"],
        data["conf"],
    ):
        text = text.strip()

        try:
            conf = float(conf)
        except ValueError:
            continue

        if text and conf >= 0:
            words.append(text)
            confs.append(conf)

    result = " ".join(words)

    if confs:
        mean_conf = sum(confs) / len(confs)
    else:
        mean_conf = 0.0

    return result, mean_conf


def orientation_score(
    recognized,
    confidence,
    target,
):
    recognized = normalize_text(
        recognized
    )

    target = normalize_text(
        target
    )

    if not recognized or not target:
        similarity = 0.0
    else:
        similarity = ratio(
            recognized,
            target
        ) / 100.0

    confidence = confidence / 100.0

    # Главное — совпадение с известной
    # TextOCR строкой.
    # Confidence используется слабее.
    score = (
        0.90 * similarity
        + 0.10 * confidence
    )

    return score, similarity


def get_raw_crop(
    dataset,
    sample_idx,
):

    filename, bbox, text = (
        dataset.samples[sample_idx]
    )

    image_path = (
        dataset.images_root
        / filename
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    x, y, w, h = bbox

    left = max(
        0,
        int(x)
    )

    top = max(
        0,
        int(y)
    )

    right = min(
        image.width,
        int(x + w)
    )

    bottom = min(
        image.height,
        int(y + h)
    )

    crop = image.crop(
        (
            left,
            top,
            right,
            bottom,
        )
    )

    return (
        crop,
        text,
        filename,
    )


def main():

    dataset = TextOCRDataset(
        annotation_path=ANNOTATIONS,
        images_root=IMAGES_ROOT,
        transform=None,
        train=False,
        limit=None,
        seed=123,
    )

    rng = random.Random(SEED)

    indices = rng.sample(
        range(len(dataset.samples)),
        N,
    )

    rows = []

    for i, sample_idx in enumerate(
        indices,
        start=1,
    ):

        crop, target, filename = (
            get_raw_crop(
                dataset,
                sample_idx,
            )
        )

        crop180 = crop.rotate(180)

        # ---------------------
        # ORIGINAL
        # ---------------------

        text0, conf0 = (
            run_tesseract(crop)
        )

        score0, sim0 = (
            orientation_score(
                text0,
                conf0,
                target,
            )
        )

        # ---------------------
        # 180°
        # ---------------------

        text180, conf180 = (
            run_tesseract(crop180)
        )

        score180, sim180 = (
            orientation_score(
                text180,
                conf180,
                target,
            )
        )

        # ---------------------
        # ORIENTATION
        # ---------------------

        original_is_180 = (
            score180 > score0
        )

        margin = abs(
            score180 - score0
        )

        rows.append({
            "sample_idx":
                sample_idx,

            "filename":
                filename,

            "target":
                target,

            "ocr_0":
                text0,

            "ocr_180":
                text180,

            "sim_0":
                sim0,

            "sim_180":
                sim180,

            "conf_0":
                conf0,

            "conf_180":
                conf180,

            "score_0":
                score0,

            "score_180":
                score180,

            "original_is_180":
                int(original_is_180),

            "margin":
                margin,
        })

        if i % 20 == 0:
            print(
                f"{i}/{N}"
            )

    df = pd.DataFrame(rows)

    # Самые уверенные решения Tesseract
    df = df.sort_values(
        "margin",
        ascending=False,
    )

    df.to_csv(
        "tesseract_orientation_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        df[
            [
                "target",
                "ocr_0",
                "ocr_180",
                "score_0",
                "score_180",
                "original_is_180",
                "margin",
            ]
        ].head(30).to_string(
            index=False
        )
    )

    print()
    print(
        "Predicted original 180°:",
        df["original_is_180"].mean()
    )

    print(
        "High-confidence decisions "
        "(margin >= 0.3):",
        (
            df["margin"] >= 0.3
        ).mean()
    )

    print()
    print(
        "Saved: "
        "tesseract_orientation_test.csv"
    )


if __name__ == "__main__":
    main()