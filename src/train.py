"""
Training and evaluation pipeline for TinyTB-Net on Chest X-Rays.
Tracks medical diagnostic metrics (Sensitivity/Recall, Specificity, ROC-AUC)
and saves the best model checkpoint.
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix

from src.dataset import create_dataloaders
from src.model import TinyTBNet, count_parameters

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_tinytb_net.pth")

def calculate_metrics(y_true, y_probs, threshold=0.5):
    """
    Computes key diagnostic metrics for medical binary classification:
    - Accuracy
    - Sensitivity (Recall of positive TB cases)
    - Specificity (Recall of normal cases)
    - ROC-AUC
    """
    y_preds = (y_probs >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_preds))
    
    # Sensitivity = TP / (TP + FN)
    sensitivity = float(recall_score(y_true, y_preds, zero_division=0))
    
    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(y_true, y_preds, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    try:
        auc = float(roc_auc_score(y_true, y_probs))
    except ValueError:
        auc = 0.5

    return {
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "roc_auc": auc,
        "confusion_matrix": (int(tn), int(fp), int(fn), int(tp))
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_targets = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs)
        all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = calculate_metrics(np.array(all_targets), np.array(all_probs))
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = calculate_metrics(np.array(all_targets), np.array(all_probs))
    metrics["loss"] = avg_loss
    metrics["probs"] = np.array(all_probs)
    metrics["targets"] = np.array(all_targets)
    return metrics


def plot_training_history(history, save_path=os.path.join(REPORTS_DIR, "training_curves.png")):
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Loss plot
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, history["val_loss"], "r--s", label="Val Loss", markersize=4)
    ax1.set_title("BCE Loss across Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Diagnostic Metrics plot
    ax2.plot(epochs, history["val_auc"], "g-^", label="Val ROC-AUC", markersize=4)
    ax2.plot(epochs, history["val_sensitivity"], "m-d", label="Val Sensitivity (Recall)", markersize=4)
    ax2.plot(epochs, history["val_accuracy"], "c--x", label="Val Accuracy", markersize=4)
    ax2.set_title("Validation Medical Metrics", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score [0.0 - 1.0]")
    ax2.set_ylim([0.0, 1.05])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to: {save_path}")


def run_training(epochs=40, lr=0.001, batch_size=16, pos_weight_mult=2.2, use_se=True):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    loaders = create_dataloaders(batch_size=batch_size)
    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    test_loader = loaders["test_loader"]

    # Compute positive weight for imbalanced loss: (num_normal / num_tb) * multiplier
    num_normal = (loaders["train_df"]["label"] == 0).sum()
    num_tb = (loaders["train_df"]["label"] == 1).sum()
    base_pos_weight = num_normal / num_tb
    effective_pos_weight = base_pos_weight * pos_weight_mult
    pos_weight = torch.tensor([effective_pos_weight], device=device)
    print(f"Training class distribution: Normal={num_normal}, TB={num_tb}")
    print(f"Clinical pos_weight: {pos_weight.item():.2f} (base {base_pos_weight:.2f} x {pos_weight_mult:.1f}x multiplier to prioritize TB detection)")

    model = TinyTBNet(use_se=use_se).to(device)
    total_params = count_parameters(model)
    print(f"Model: TinyTB-Net (use_se={use_se}) | Total params: {total_params:,} (~{total_params/1024:.1f} KB INT8)")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [], "val_loss": [],
        "val_accuracy": [], "val_auc": [], "val_sensitivity": []
    }

    best_val_auc = 0.0

    print("\nStarting Training Walkthrough...")
    print("-" * 75)
    print(f"{'Epoch':<6} | {'Train Loss':<11} | {'Val Loss':<9} | {'Val Acc':<8} | {'Sensitivity':<12} | {'Val AUC':<8} | {'Time'}")
    print("-" * 75)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["val_accuracy"].append(val_m["accuracy"])
        history["val_auc"].append(val_m["roc_auc"])
        history["val_sensitivity"].append(val_m["sensitivity"])

        print(f"{epoch:<6} | {train_m['loss']:<11.4f} | {val_m['loss']:<9.4f} | {val_m['accuracy']:<8.2%} | {val_m['sensitivity']:<12.2%} | {val_m['roc_auc']:<8.4f} | {elapsed:.1f}s")

        # Save checkpoint with highest ROC-AUC
        if val_m["roc_auc"] > best_val_auc:
            best_val_auc = float(val_m["roc_auc"])
            clean_val_metrics = {
                "accuracy": float(val_m["accuracy"]),
                "sensitivity": float(val_m["sensitivity"]),
                "specificity": float(val_m["specificity"]),
                "roc_auc": float(val_m["roc_auc"]),
                "confusion_matrix": tuple(int(x) for x in val_m["confusion_matrix"]),
                "loss": float(val_m["loss"]),
            }
            torch.save({
                "epoch": int(epoch),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_val_auc,
                "val_metrics": clean_val_metrics,
                "arch": "TinyTBNet",
                "use_se": use_se,
                "num_params": int(total_params),
            }, BEST_MODEL_PATH)

    print("-" * 75)
    print(f"Best Validation ROC-AUC: {best_val_auc:.4f} (Saved to {BEST_MODEL_PATH})")

    plot_training_history(history)

    # Final Evaluation on the unseen Test Set
    print("\nRunning Evaluation on Unseen Test Set...")
    checkpoint = torch.load(BEST_MODEL_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_m = evaluate(model, test_loader, criterion, device)

    tn, fp, fn, tp = test_m["confusion_matrix"]
    print(f"\nFinal Test Results (Standard Threshold = 0.50):")
    print(f"  Test Accuracy:     {test_m['accuracy']:.2%}")
    print(f"  Test Sensitivity:  {test_m['sensitivity']:.2%} (Caught {tp}/{tp+fn} actual TB cases)")
    print(f"  Test Specificity:  {test_m['specificity']:.2%} (Cleared {tn}/{tn+fp} healthy lungs)")
    print(f"  Test ROC-AUC:      {test_m['roc_auc']:.4f}")
    print(f"  Confusion Matrix:  TN={tn}, FP={fp}, FN={fn}, TP={tp}\n")

    # Multi-threshold clinical operating point analysis
    print("Clinical Operating Curve Analysis (Sensitivity vs Specificity Trade-off):")
    for th in [0.35, 0.40, 0.45, 0.50]:
        th_m = calculate_metrics(test_m["targets"], test_m["probs"], threshold=th)
        c_tn, c_fp, c_fn, c_tp = th_m["confusion_matrix"]
        print(f"  Threshold {th:.2f} -> Sensitivity: {th_m['sensitivity']:.2%} (Caught {c_tp}/{c_tp+c_fn}), Specificity: {th_m['specificity']:.2%} (Cleared {c_tn}/{c_tn+c_fp}), Acc: {th_m['accuracy']:.2%}")

    return test_m


if __name__ == "__main__":
    run_training(epochs=40, batch_size=16, pos_weight_mult=2.2, use_se=True)
