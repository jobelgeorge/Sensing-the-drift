#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training entrypoint for the ImprovedLSTMModel.
- Loads data
- Trains with frequency-weighted loss and mixup
- Saves: best model checkpoint, training plots, metrics JSON/CSV, and predictions
Everything is saved under the script directory (ROOT).
"""

from pathlib import Path
import os
import sys
import argparse
import json
import time
import math
import traceback

# ---------- Project root + relative path helper (from your snippet) ----------
def _here():
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()

ROOT = _here()

def R(p) -> Path:
    """Resolve a path `p` relative to the project ROOT."""
    return (ROOT / Path(p)).resolve()

# Ensure all relative I/O happens under the project root
os.chdir(ROOT)

# ---------- Ensure imports work no matter where main.py runs ----------
def find_project_root():
    here = Path.cwd()
    # Prefer a directory that has both 'utils' and 'models'
    for p in [here, *here.parents]:
        if (p / "utils").exists() and (p / "models").exists():
            return p
    # Fallback: look for common markers (git, requirements, pyproject)
    for p in [here, *here.parents]:
        if any((p / m).exists() for m in (".git", "requirements.txt", "pyproject.toml")):
            return p
    return here  # last resort

PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------- Imports ----------
# Use a non-interactive backend for matplotlib (so we can save figures headlessly)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import numpy as np
import torch.nn as nn
import pandas as pd

from utils.reproducibility_utils import seed_everything
from utils.data_loading import load_data, load_test_data, feature_names
from models.ImprovedLSTMModel import ImprovedLSTMModel
from utils.FrequencyBasedWeightedLoss import FrequencyBasedWeightedLoss
from utils.ImbalancedSampling import MixupAugmentation

# ---------- CLI ----------
def build_argparser():
    p = argparse.ArgumentParser(description="Train ImprovedLSTMModel and save artifacts in ROOT.")
    # Data
    p.add_argument("--train_dir", type=str, default="datasets_challenge/train", help="Train data directory (relative to ROOT or absolute).")
    p.add_argument("--test_dir", type=str, default="datasets_challenge/test", help="Test data directory (relative to ROOT or absolute).")
    p.add_argument("--train_ratio", type=float, default=0.8, help="Train/val split ratio.")
    p.add_argument("--batchsize", type=int, default=256, help="Batch size.")
    p.add_argument("--sequence_length", type=int, default=20, help="Sequence length.")
    p.add_argument("--columns_to_drop", type=str, nargs="*", default=["sideSlip","vyCG","time","longitude","latitude"], help="Columns to drop.")
    p.add_argument("--threshold", action="store_true", default=True, help="Enable thresholding on a column.")
    p.add_argument("--threshold_value", type=float, default=3.0, help="Threshold value.")
    p.add_argument("--threshold_column", type=str, default="vxCG", help="Which column to threshold on.")
    p.add_argument("--target", type=str, default="sideSlip", help="Target column.")
    p.add_argument("--scaling", action="store_true", default=True, help="Enable scaling.")

    # Model
    p.add_argument("--rnn_units", type=int, default=128)
    p.add_argument("--fc_units", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--use_residual", action="store_true", default=True)
    p.add_argument("--use_attention", action="store_true", default=True)
    p.add_argument("--predict_uncertainty", action="store_true", default=False)

    # Training
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--early_stop", type=int, default=25)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)

    # Loss
    p.add_argument("--percentiles", type=float, nargs=2, default=(90.0, 99.0))
    p.add_argument("--base_weights", type=float, nargs=3, default=(1.0, 2.0, 5.0))
    p.add_argument("--use_huber", action="store_true", default=True)
    p.add_argument("--huber_beta", type=float, default=0.1)
    p.add_argument("--focal_gamma", type=float, default=0.5)

    # Output names
    p.add_argument("--model_name", type=str, default="ImprovedLSTMModel", help="Used in filenames.")
    p.add_argument("--checkpoint_name", type=str, default=None, help="Optional custom checkpoint filename.")
    p.add_argument("--run_tag", type=str, default=None, help="Optional tag appended to output files.")

    return p

# ---------- Utilities ----------
def _to_rel_or_abs(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else R(p)

def _format_tag(args):
    if args.run_tag:
        return f"_{args.run_tag}"
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"_{ts}"

def save_json(obj, path: Path):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

# ---------- Main Training ----------
def main():
    args = build_argparser().parse_args()

    # Paths
    train_dir = _to_rel_or_abs(args.train_dir)
    test_dir  = _to_rel_or_abs(args.test_dir)

    # Matplotlib style (match your notebook)
    try:
        plt.style.use("seaborn-v0_8")
    except Exception:
        pass

    # Reproducibility
    seed_everything(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data
    print("Loading train/val data...")
    train_dataloader, val_dataloader, pipeline = load_data(
        train_path=str(train_dir),
        train_ratio=args.train_ratio,
        batchsize=args.batchsize,
        columns_to_drop=args.columns_to_drop,
        sequence_length=args.sequence_length,
        threshold=args.threshold,
        threshold_value=args.threshold_value,
        threshold_column=args.threshold_column,
        target=args.target,
        scaling=args.scaling,
        seed=args.seed,
    )

    # Gather training targets for loss init
    print("Extracting training targets for frequency-based loss...")
    train_dataset = train_dataloader.dataset
    all_train_targets = []
    for i in range(len(train_dataset)):
        _, y = train_dataset[i]
        # y may be tensor shape (), ensure scalar
        all_train_targets.append(float(y))
    y_train_np = np.array(all_train_targets, dtype=np.float32)
    print(f"Collected {len(y_train_np)} training samples")
    if len(y_train_np) > 0:
        print(f"Target range: [{y_train_np.min():.4f}, {y_train_np.max():.4f}]")
        print(f"Target mean: {y_train_np.mean():.4f}, std: {y_train_np.std():.4f}")

    # Peek first batch to infer input size
    first_batch = next(iter(train_dataloader))
    X0, y0 = first_batch
    input_size = X0.shape[2]
    output_size = 1

    # Model
    model = ImprovedLSTMModel(
        input_size=input_size,
        output_size=output_size,
        rnn_units=args.rnn_units,
        num_layers=args.num_layers,
        fc_units=args.fc_units,
        dropout=args.dropout,
        use_residual=args.use_residual,
        use_attention=args.use_attention,
        predict_uncertainty=args.predict_uncertainty,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.model_name}")
    print(f"Parameters: {total_params:,} ({trainable_params:,} trainable)")
    print(f"Size (fp32): {total_params * 4 / 1024**2:.2f} MB")

    # Loss
    criterion = FrequencyBasedWeightedLoss(
        train_targets=y_train_np,
        percentiles=tuple(args.percentiles),
        base_weights=tuple(args.base_weights),
        use_huber=args.use_huber,
        huber_beta=args.huber_beta,
        focal_gamma=args.focal_gamma,
    )
    print("Loss configuration set.")

    # Mixup
    mixup = MixupAugmentation(alpha=0.15, prob=0.3, prioritize_rare=False, rare_threshold=None)
    print("Mixup: alpha=0.15, prob=0.3, prioritize_rare=False")

    # Optim + LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # History
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
        "lr": [],
    }

    best = {"loss": math.inf, "epoch": -1}
    model_tag = _format_tag(args)
    ckpt_name = args.checkpoint_name or f"best_{args.model_name.lower()}{model_tag}.pth"

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_mae  = 0.0

        for step, (x, y) in enumerate(train_dataloader):
            Xb = x.to(device)
            yb = y.to(device).reshape([len(y), 1])

            # Mixup
            Xb, yb = mixup(Xb, yb)

            optimizer.zero_grad(set_to_none=True)
            pred = model(Xb)
            loss = criterion(pred, yb)
            mae = torch.mean(torch.abs(pred - yb))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item())
            running_mae  += float(mae.item())

        avg_train_loss = running_loss / (step + 1)
        avg_train_mae  = running_mae  / (step + 1)

        # Validation
        model.eval()
        val_loss_acc = 0.0
        val_mae_acc  = 0.0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for vstep, (x, y) in enumerate(val_dataloader):
                Xv = x.to(device)
                yv = y.to(device).reshape([len(y), 1])
                pv = model(Xv)
                vloss = criterion(pv, yv)
                vmae  = torch.mean(torch.abs(pv - yv))

                val_loss_acc += float(vloss.item())
                val_mae_acc  += float(vmae.item())

                all_val_preds.append(pv)
                all_val_targets.append(yv)

        avg_val_loss = val_loss_acc / (vstep + 1)
        avg_val_mae  = val_mae_acc  / (vstep + 1)

        # Bucket metrics if available
        bucket_line = ""
        try:
            if hasattr(criterion, "bucket_metrics"):
                vp = torch.cat(all_val_preds, dim=0)
                vt = torch.cat(all_val_targets, dim=0)
                bucket_stats = criterion.bucket_metrics(vp, vt)
                bucket_line = f" | Many: {bucket_stats.get('mae_many', float('nan')):.4f} | Med: {bucket_stats.get('mae_medium', float('nan')):.4f} | Few: {bucket_stats.get('mae_few', float('nan')):.4f}"
        except Exception as e:
            bucket_line = ""

        # Record
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_mae"].append(avg_train_mae)
        history["val_mae"].append(avg_val_mae)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        # Scheduler step after epoch
        scheduler.step()

        # Progress
        print(f"Epoch {epoch+1:03d}/{args.epochs} | Train Loss {avg_train_loss:.5f} | Val Loss {avg_val_loss:.5f} | "
              f"Train MAE {avg_train_mae:.5f} | Val MAE {avg_val_mae:.5f} | LR {history['lr'][-1]:.6f}{bucket_line}")

        # Checkpoint
        if avg_val_loss < best["loss"]:
            best.update({"loss": avg_val_loss, "epoch": epoch})
            torch.save({
                "epoch": epoch,
                "model_type": args.model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_mae": avg_train_mae,
                "val_mae": avg_val_mae,
                "config": vars(args),
            }, R(ckpt_name))
            print(f"  -> New best checkpoint saved: {ckpt_name}")

        # Early stopping
        if best["epoch"] != -1 and (epoch - best["epoch"]) > args.early_stop:
            print(f"Early stopping at epoch {epoch+1} (best: {best['epoch']+1})")
            break

    # Load best
    best_epoch = best["epoch"]
    if best_epoch != -1:
        ckpt = torch.load(R(ckpt_name), map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Best model loaded: epoch {best_epoch+1} | best val loss {best['loss']:.5f}")
    else:
        print("No improvement found; using last model state.")

    # ---- Save training curves ----
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        # Loss curves
        axes[0, 0].plot(history["train_loss"], label="Train", linewidth=2)
        axes[0, 0].plot(history["val_loss"], label="Val", linewidth=2)
        if best_epoch >= 0:
            axes[0, 0].axvline(x=best_epoch, color="red", linestyle="--", linewidth=2, label="Best")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # MAE curves
        axes[0, 1].plot(history["train_mae"], label="Train MAE", linewidth=2)
        axes[0, 1].plot(history["val_mae"], label="Val MAE", linewidth=2)
        if best_epoch >= 0:
            axes[0, 1].axvline(x=best_epoch, color="red", linestyle="--", linewidth=2, label="Best")
        axes[0, 1].set_title("Mean Absolute Error")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("MAE")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # LR schedule
        axes[1, 0].plot(history["lr"], linewidth=2, marker="o", markersize=3)
        if best_epoch >= 0:
            axes[1, 0].axvline(x=best_epoch, color="red", linestyle="--", linewidth=2)
        axes[1, 0].set_title("Learning Rate")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("LR")
        axes[1, 0].set_yscale("log")
        axes[1, 0].grid(True, alpha=0.3)

        # Generalization gap
        axes[1, 1].plot(history["train_loss"], label="Train", linewidth=2)
        axes[1, 1].plot(history["val_loss"], label="Val", linewidth=2)
        if best_epoch >= 0:
            axes[1, 1].axvline(x=best_epoch, color="red", linestyle="--", linewidth=2, label="Best")
        axes[1, 1].set_title("Generalization Gap")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()
        fig_out = R(f"training_plots{model_tag}.png")
        plt.savefig(fig_out, dpi=150)
        plt.close(fig)
        print(f"Saved training plots to {fig_out.name}")
    except Exception as e:
        print("Failed to save training plots:", e)

    # ---- Training history JSON/CSV ----
    try:
        save_json({
            "best_epoch": best_epoch,
            "best_val_loss": best["loss"],
            "history": history,
        }, R(f"metrics{model_tag}.json"))
        hist_df = pd.DataFrame(history)
        hist_df.index.name = "epoch"
        hist_df.to_csv(R(f"metrics{model_tag}.csv"))
        print("Saved metrics JSON/CSV.")
    except Exception as e:
        print("Failed to save metrics:", e)

    # ---- Train set visualization (truth vs pred) ----
    try:
        print("Preparing train-set truth vs pred visualization...")
        train_dataloader_unshuffled = load_test_data(
            test_path=str(train_dir),
            batchsize=args.batchsize,
            pipeline=pipeline,
            load_anonymized_data=False,
        )

        model.eval()
        preds = []
        truths = []
        with torch.no_grad():
            for step, (x, y) in enumerate(train_dataloader_unshuffled):
                Xb = x.to(device)
                yb = y.to(device).reshape([len(y), 1])
                pb = model(Xb)
                preds.append(pb.reshape(-1).cpu().numpy())
                truths.append(yb.reshape(-1).cpu().numpy())
        preds = np.concatenate(preds)
        truths = np.concatenate(truths)

        # Save simple matplotlib version (no plotly dependency)
        plt.figure(figsize=(16, 5))
        plt.plot(truths, label="truth")
        plt.plot(preds, label="pred", alpha=0.8)
        plt.title("Ground Truth vs Predictions (Train)")
        plt.xlabel("Sample Index")
        plt.ylabel("Sideslip")
        plt.legend()
        out_png = R(f"train_truth_vs_pred{model_tag}.png")
        plt.savefig(out_png, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"Saved train truth-vs-pred plot to {out_png.name}")
    except Exception as e:
        print("Failed train-set visualization:", e)

    # ---- Test predictions ----
    try:
        print("Loading test data...")
        test_dataloader = load_test_data(
            test_path=str(test_dir),
            batchsize=args.batchsize,
            pipeline=pipeline,
            load_anonymized_data=True,
        )

        model.eval()
        test_preds = []
        with torch.no_grad():
            for step, (x, y) in enumerate(test_dataloader):
                Xb = x.to(device)
                pb = model(Xb)
                test_preds.append(pb.reshape(-1).cpu().numpy())
        test_preds = np.concatenate(test_preds)
        print("Shape of predictions:", test_preds.shape)

        y_pred = pd.DataFrame({"predicted_sideslip": test_preds})
        y_pred["id"] = np.arange(len(test_preds))
        pred_csv = R(f"predictions{model_tag}.csv")
        y_pred.to_csv(pred_csv, index=False)
        print(f"✓ Predictions saved to '{pred_csv.name}'")
    except Exception as e:
        print("Failed to generate test predictions:", e)

    # ---- Final summary ----
    print("\nTraining Summary")
    if len(history["val_loss"]) > 0:
        print(f"Best Epoch: {best_epoch + 1}/{len(history['val_loss'])}")
        print(f"Best Val Loss: {min(history['val_loss']):.5f}")
        print(f"Best Val MAE: {min(history['val_mae']):.5f}")
        print(f"Final LR: {history['lr'][-1]:.6f}")
        print(f"Early Stop: {'Yes' if len(history['val_loss']) < args.epochs else 'No'}")
    else:
        print("No epochs recorded.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error in main():", e)
        traceback.print_exc()
        sys.exit(1)
