import argparse
import random
from pathlib import Path

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

    print(f"Loaded checkpoint: {path}")


@torch.no_grad()
def validate(model, loader, device, use_bf16):
    model.eval()

    total = 0

    normal_correct = 0
    normal_brier_sum = 0.0

    sym_correct = 0
    sym_brier_sum = 0.0

    for images, labels in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

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

        # symmetry-aware inference
        sym_logits = (
            logits - logits_rot
        ) / 2.0

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
        "--checkpoint",
        default="best_mjsynth_synthtiger.pth",
    )

    parser.add_argument(
        "--output",
        default="best_mjsynth_synthtiger_sym.pth",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
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

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_bf16 = (
        device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    print("Device:", device)
    print("BF16:", use_bf16)

    samples = load_synthtiger_samples(args.data)

    train_samples, val_samples = split_samples(
        samples,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    print(f"All valid:         {len(samples):,}")
    print(f"Train base images: {len(train_samples):,}")
    print(f"Val base images:   {len(val_samples):,}")
    print(f"Val examples:      {len(val_samples) * 2:,}")

    train_dataset = SynthTigerDataset(
        train_samples,
        train=True,
    )

    val_dataset = SynthTigerDataset(
        val_samples,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    model = OrientationModel().to(device)

    load_checkpoint(
        model,
        args.checkpoint,
        device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_brier = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        seen = 0

        for batch_idx, (images, labels) in enumerate(
            train_loader,
            start=1,
        ):
            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            rotated = torch.rot90(
                images,
                k=2,
                dims=(-2, -1),
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                # Symmetry-aware training.
                # We optimize the same quantity that will be used during inference.
                # Rotating an image by 180 degrees must invert its orientation label,
                # therefore the ideal logits should satisfy:
                #     f(rot180(x)) = -f(x)

                logits = model(images)
                logits_rot = model(rotated)
                sym_logits = (logits - logits_rot) / 2.0

                loss = criterion(
                    sym_logits,
                    labels,
                )

            loss.backward()
            optimizer.step()

            bs = labels.size(0)

            running_loss += (
                loss.item() * bs
            )

            seen += bs

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} "
                    f"| batch {batch_idx}/{len(train_loader)} "
                    f"| sym_loss="
                    f"{running_loss / seen:.6f}"
                )

        train_loss = running_loss / seen

        metrics = validate(
            model,
            val_loader,
            device,
            use_bf16,
        )

        print()
        print(f"Epoch {epoch}/{args.epochs}")
        print(
            f"sym_train_loss={train_loss:.6f}"
        )

        print(
            "NORMAL "
            f"acc={metrics['normal_acc']:.6f} "
            f"Brier={metrics['normal_brier']:.6f} "
            f"score={1 - metrics['normal_brier']:.6f}"
        )

        print(
            "ROT-CONSISTENCY "
            f"acc={metrics['sym_acc']:.6f} "
            f"Brier={metrics['sym_brier']:.6f} "
            f"score={1 - metrics['sym_brier']:.6f}"
        )

        if metrics["sym_brier"] < best_brier:
            best_brier = metrics["sym_brier"]

            torch.save(
                model.state_dict(),
                args.output,
            )

            print(
                f"Saved BEST -> {args.output}"
            )

        last_path = (
            Path(args.output)
            .with_suffix("")
            .as_posix()
            + "_last.pth"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "metrics": metrics,
            },
            last_path,
        )

        print()

    print(
        f"Finished. Best sym Brier: "
        f"{best_brier:.6f}"
    )


if __name__ == "__main__":
    main()