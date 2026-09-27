import argparse
import gc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import OrientationModel
from synthtiger_dataset import (
    load_synthtiger_samples,
    split_samples,
    SynthTigerDataset,
)


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device)

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


@torch.no_grad()
def validate(model, loader, device, use_bf16):
    """
    Evaluate both standard predictions and rotation-consistent predictions.
    Returns accuracy and Brier score for both inference modes.
    """
    model.eval()

    total = 0

    normal_correct = 0
    normal_brier_sum = 0.0

    sym_correct = 0
    sym_brier_sum = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)

        labels = labels.to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            logits = model(images)

            rotated = torch.rot90(
                images,
                k=2,
                dims=(-2, -1),
            )

            logits_rot = model(rotated)

        logits = logits.float()
        logits_rot = logits_rot.float()

        # normal
        probs = torch.sigmoid(logits)

        normal_brier_sum += (
            (probs - labels) ** 2
        ).sum().item()

        normal_correct += (
            (probs >= 0.5) == labels.bool()
        ).sum().item()

        # symmetry inference
        sym_logits = (logits - logits_rot) / 2.0
        sym_probs = torch.sigmoid(sym_logits)

        sym_brier_sum += (
            (sym_probs - labels) ** 2
        ).sum().item()

        sym_correct += (
            (sym_probs >= 0.5) == labels.bool()
        ).sum().item()

        total += labels.numel()

    return {
        "normal_acc": normal_correct / total,
        "normal_brier": normal_brier_sum / total,
        "sym_acc": sym_correct / total,
        "sym_brier": sym_brier_sum / total,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/synthtiger_train",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_bf16 = (
        device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    print("Device:", device)
    print("BF16:", use_bf16)

    print("Reading samples...")

    samples = load_synthtiger_samples(args.data)

    # Use the same deterministic split for every checkpoint
    # so model comparisons are directly comparable.

    _, val_samples = split_samples(
        samples,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    val_dataset = SynthTigerDataset(
        val_samples,
        train=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    checkpoints = [
        (
            "MJSynth original",
            "best_mobilenetv3_orientation.pth",
        ),
        (
            "TextOCR original",
            "best_textocr_orientation.pth",
        ),
        (
            "MJSynth -> SynthTIGER",
            "best_mjsynth_synthtiger.pth",
        ),
        (
            "TextOCR -> SynthTIGER",
            "best_textocr_synthtiger.pth",
        ),
    ]

    results = []

    for name, checkpoint_path in checkpoints:
        print()
        print("=" * 70)
        print(name)
        print(checkpoint_path)

        model = OrientationModel().to(device)

        load_checkpoint(
            model,
            checkpoint_path,
            device,
        )

        metrics = validate(
            model,
            val_loader,
            device,
            use_bf16,
        )

        results.append((name, metrics))

        print(
            f"NORMAL: "
            f"acc={metrics['normal_acc']:.6f} "
            f"Brier={metrics['normal_brier']:.6f} "
            f"score={1 - metrics['normal_brier']:.6f}"
        )

        print(
            f"SYM:    "
            f"acc={metrics['sym_acc']:.6f} "
            f"Brier={metrics['sym_brier']:.6f} "
            f"score={1 - metrics['sym_brier']:.6f}"
        )

        del model
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print()
    print("=" * 92)
    print(
        f"{'MODEL':30s} "
        f"{'NORMAL ACC':>12s} "
        f"{'NORMAL BRIER':>14s} "
        f"{'SYM ACC':>12s} "
        f"{'SYM BRIER':>12s}"
    )
    print("=" * 92)

    for name, m in results:
        print(
            f"{name:30s} "
            f"{m['normal_acc']:12.6f} "
            f"{m['normal_brier']:14.6f} "
            f"{m['sym_acc']:12.6f} "
            f"{m['sym_brier']:12.6f}"
        )


if __name__ == "__main__":
    main()