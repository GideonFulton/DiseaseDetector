import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from PIL import Image

from utils import (
    project_root,
    data_root_color,
    list_all_classes,
    crop_name_from_class,
    train_val_split,
    set_seed,
    list_crop_types
)

IMG_EXTS = (".jpg", ".jpeg", ".png")


# Training settings for each crop-specific disease model
@dataclass
class DiseaseTrainConfig:
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 10
    lr: float = 1e-3
    seed: int = 42
    val_frac: float = 0.15
    max_train: int = 4000
    max_val: int = 1000


# Use MPS on Mac if available, otherwise fall back to CPU
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Make crop names safe to use in checkpoint filenames
def safe_crop_name(crop_name: str) -> str:
    return (
        crop_name.lower()
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace(" ", "_")
    )


# Standard checkpoint path for a crop-specific disease model
def default_ckpt_path_for_crop(crop_name: str) -> str:
    return os.path.join(project_root(), "artifacts", f"{safe_crop_name(crop_name)}_mobilenetv2.pt")


# Slightly cleaner crop name for plots/titles
def pretty_crop_name(name: str) -> str:
    return name.replace("_(maize)", "").replace(",", "").replace("_", " ")


# Return all dataset class folders that belong to one crop
def list_classes_for_crop(root_dir: str, crop_name: str) -> List[str]:
    out = []
    for class_name in list_all_classes(root_dir):
        if crop_name_from_class(class_name) == crop_name:
            out.append(class_name)
    return sorted(out)


# Convert dataset folder labels into more readable disease labels
def pretty_disease_label(class_name: str, crop_name: str) -> str:
    healthy_name = f"{crop_name}___healthy"
    if class_name == healthy_name:
        return "No disease (healthy)"
    prefix = f"{crop_name}___"
    return class_name.replace(prefix, "").replace("_", " ")


# Ground-truth class comes from the parent folder name
def ground_truth_class_from_path(image_path: str) -> str:
    return os.path.basename(os.path.dirname(image_path))


# Build train/validation subsets for one crop's disease classes
def build_crop_disease_subsets(
    root: str,
    crop_name: str,
    cfg: DiseaseTrainConfig,
):
    tf_train = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
    ])

    base_train = datasets.ImageFolder(root, transform=tf_train)
    base_val = datasets.ImageFolder(root, transform=tf_val)

    crop_classes = list_classes_for_crop(root, crop_name)
    if not crop_classes:
        raise RuntimeError(f"No class folders found for crop '{crop_name}' in {root}")

    crop_old_idxs = sorted(
        [base_train.class_to_idx[c] for c in crop_classes if c in base_train.class_to_idx]
    )
    crop_old_set = set(crop_old_idxs)

    crop_indices = [
        i for i, (_, y) in enumerate(base_train.samples)
        if y in crop_old_set
    ]
    if not crop_indices:
        raise RuntimeError(f"No images found for crop '{crop_name}' after filtering")

    train_idx, val_idx = train_val_split(crop_indices, val_frac=cfg.val_frac, seed=cfg.seed)
    train_idx = train_idx[:cfg.max_train]
    val_idx = val_idx[:cfg.max_val]

    train_ds = Subset(base_train, train_idx)
    val_ds = Subset(base_val, val_idx)

    # Remap original dataset class indices to a compact 0...K-1 range
    remap = {old: new for new, old in enumerate(crop_old_idxs)}

    return base_train, train_ds, val_ds, crop_old_idxs, remap


# Apply the crop-specific remapping to a label batch
def remap_labels(y: torch.Tensor, remap: Dict[int, int]) -> torch.Tensor:
    y2 = torch.empty_like(y)
    for i in range(len(y)):
        y2[i] = remap[int(y[i].item())]
    return y2


# Train one disease model for one crop
def train_crop_disease_model(crop_name: str, out_path: str, cfg: DiseaseTrainConfig) -> None:
    set_seed(cfg.seed)
    device = get_device()
    print("Device:", device)

    root = data_root_color()
    print("Dataset root:", root)
    print("Crop:", crop_name)

    _, train_ds, val_ds, crop_old_idxs, remap = build_crop_disease_subsets(root, crop_name, cfg)

    num_classes = len(remap)
    print("Disease classes:", num_classes)
    print("Train images:", len(train_ds))
    print("Val images:", len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    # Fine-tune only the last block and classifier
    for p in model.parameters():
        p.requires_grad = True

    for name, p in model.named_parameters():
        if not name.startswith("features.17") and not name.startswith("classifier"):
            p.requires_grad = False

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr
    )

    for epoch in range(cfg.epochs):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0

        for x, y in train_loader:
            x = x.to(device)
            y = remap_labels(y, remap).to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += x.size(0)

        train_loss = total_loss / max(1, seen)
        train_acc = correct / max(1, seen)

        model.eval()
        v_correct, v_seen = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = remap_labels(y, remap).to(device)
                logits = model(x)
                v_correct += (logits.argmax(1) == y).sum().item()
                v_seen += x.size(0)

        val_acc = v_correct / max(1, v_seen)

        print(
            f"Epoch {epoch+1}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
        )

    idx_to_class = {v: k for k, v in datasets.ImageFolder(root).class_to_idx.items()}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "crop_name": crop_name,
            "crop_old_class_idxs": crop_old_idxs,
            "remap": remap,
            "image_size": cfg.image_size,
            "arch": "mobilenet_v2",
            "idx_to_class": idx_to_class,
        },
        out_path,
    )
    print("Saved model to:", out_path)


# Train and save one disease model for each crop in the dataset
def train_all_crop_disease_models(
    crops: List[str] = None,
    cfg: DiseaseTrainConfig = None,
    skip_existing: bool = False,
) -> None:
    """
    Train and save one disease classifier per crop.

    If crops is None, train for every crop found in the dataset.
    If skip_existing is True, already-saved checkpoints are skipped.
    """
    if cfg is None:
        cfg = DiseaseTrainConfig()

    root = data_root_color()

    if crops is None:
        crops = list_crop_types(root)

    print("Training disease models for crops:")
    for crop in crops:
        print(" -", crop)

    for crop in crops:
        out_path = default_ckpt_path_for_crop(crop)

        if skip_existing and os.path.isfile(out_path):
            print(f"[SKIP] Checkpoint already exists for {crop}: {out_path}")
            continue

        print(f"\n[TRAIN] Crop disease model for: {crop}")
        train_crop_disease_model(crop, out_path, cfg)


# Load a saved crop-specific disease model checkpoint
def load_crop_disease_model(ckpt_path: str):
    device = get_device()
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    num_classes = len(ckpt["remap"])
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    idx_to_class = ckpt["idx_to_class"]

    new_to_old = {new: old for old, new in ckpt["remap"].items()}
    new_to_name = {new: idx_to_class[new_to_old[new]] for new in range(num_classes)}

    return model, device, ckpt["image_size"], new_to_name, ckpt["crop_name"]


# Predict just the top disease class for one image
@torch.no_grad()
def predict_crop_disease(ckpt_path: str, image_path: str) -> Tuple[str, float]:
    model, device, image_size, new_to_name, _ = load_crop_disease_model(ckpt_path)

    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred = int(torch.argmax(probs).item())

    return new_to_name[pred], float(probs[pred].item())


# Predict all disease class probabilities for one image
@torch.no_grad()
def predict_all_crop_disease_probs(ckpt_path: str, image_path: str):
    model, device, image_size, new_to_name, crop_name = load_crop_disease_model(ckpt_path)

    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()

    pred_idx = int(probs.argmax().item())
    pred_name = new_to_name[pred_idx]
    pred_prob = float(probs[pred_idx].item())

    all_results = [(new_to_name[i], float(probs[i].item())) for i in range(len(probs))]

    return img, all_results, pred_name, pred_prob, crop_name


# Show image + bar chart of disease probabilities for one crop-specific model
def visualize_crop_disease_prediction(ckpt_path: str, image_path: str):
    import matplotlib.pyplot as plt

    gt = ground_truth_class_from_path(image_path)
    img, all_results, pred_name, pred_prob, crop_name = predict_all_crop_disease_probs(ckpt_path, image_path)

    gt_pretty = pretty_disease_label(gt, crop_name)
    pred_pretty = pretty_disease_label(pred_name, crop_name)

    sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)
    labels_plain = [pretty_disease_label(name, crop_name) for name, _ in sorted_results]
    probs = [p for _, p in sorted_results]

    marked_labels = []
    for lab in labels_plain:
        prefix = ""
        if lab == gt_pretty and lab == pred_pretty:
            prefix = "✓ "
        else:
            if lab == pred_pretty:
                prefix += "PREDICTION "
            if lab == gt_pretty:
                prefix += "GROUND TRUTH "
            if prefix:
                prefix = prefix.strip() + " "
        marked_labels.append(prefix + lab)

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img)
    ax0.axis("off")
    ax0.set_title("Input image")

    summary = f"Ground Truth: {gt_pretty}\nPrediction: {pred_pretty}  (conf={pred_prob:.3f})"
    ax0.text(
        0.0, -0.08, summary,
        transform=ax0.transAxes,
        fontsize=11,
        va="top"
    )

    ax1 = fig.add_subplot(gs[0, 1])
    labels_rev = marked_labels[::-1]
    labels_plain_rev = labels_plain[::-1]
    probs_rev = probs[::-1]

    bars = ax1.barh(labels_rev, probs_rev)

    for bar, lab in zip(bars, labels_plain_rev):
        if lab == gt_pretty:
            bar.set_color("green")
        if lab == pred_pretty and pred_pretty != gt_pretty:
            bar.set_color("red")

    ax1.set_xlabel("Probability")
    ax1.set_title(f"Class probabilities ({pretty_crop_name(crop_name)} disease model)")
    ax1.set_xlim(0, 1)
    ax1.xaxis.grid(True, linestyle="--", alpha=0.6)
    ax1.set_axisbelow(True)

    for bar in bars:
        width = bar.get_width()
        ax1.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}",
            va="center",
            fontsize=9
        )

    plt.tight_layout()
    plt.show()

    print("Ground truth:", gt)
    print(f"Prediction: {pred_name} (confidence={pred_prob:.6f})")


# Evaluate one crop-specific disease model on its validation split
def evaluate_crop_disease_model(ckpt_path: str):
    model, device, image_size, new_to_name, crop_name = load_crop_disease_model(ckpt_path)

    cfg = DiseaseTrainConfig()
    root = data_root_color()

    _, _, val_ds, remap_old_to_new, class_idx_to_crop_old = None, None, None, None, None
    _, _, val_ds, crop_old_idxs, remap = build_crop_disease_subsets(root, crop_name, cfg)
    loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu()
            y_remap = remap_labels(y, remap)

            correct += (preds == y_remap).sum().item()
            total += len(y)

    acc = correct / max(total, 1)
    print(f"\n{crop_name} disease validation accuracy: {acc*100:.2f}%")
    print(f"Correct: {correct} / {total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop", type=str, default=None, help="Crop name, e.g. Tomato or Potato")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train_all", action="store_true", help="Train disease models for all crops")
    parser.add_argument("--predict", type=str, default=None, help="Path to an image")
    parser.add_argument("--viz", action="store_true", help="Show image + disease probability chart")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on crop-specific validation images")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint path. If omitted, uses ../artifacts/<crop>_mobilenetv2.pt"
    )

    args = parser.parse_args()

    if args.train_all:
        cfg = DiseaseTrainConfig()
        train_all_crop_disease_models(cfg=cfg)
        return

    if args.crop is None and not args.train_all:
        raise ValueError("You must provide --crop unless using --train_all")

    ckpt_path = args.ckpt if args.ckpt is not None else default_ckpt_path_for_crop(args.crop)

    if args.train:
        cfg = DiseaseTrainConfig()
        train_crop_disease_model(args.crop, ckpt_path, cfg)

    elif args.evaluate:
        evaluate_crop_disease_model(ckpt_path)

    elif args.predict:
        if args.viz:
            visualize_crop_disease_prediction(ckpt_path, args.predict)
        else:
            label, conf = predict_crop_disease(ckpt_path, args.predict)
            print(f"Prediction: {label} (confidence={conf:.6f})")

    else:
        print("Use --train, --train_all, --predict path/to/image.jpg, or --evaluate")


if __name__ == "__main__":
    main()