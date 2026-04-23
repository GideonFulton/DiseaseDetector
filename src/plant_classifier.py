import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from PIL import Image

from utils import (
    data_root_color,
    list_all_classes,
    crop_name_from_class,
    train_val_split,
    set_seed,
)

from crop_disease_classifier import (
    default_ckpt_path_for_crop,
    predict_crop_disease,
    visualize_crop_disease_prediction,
)

# Crop classifier used for the first stage of the pipeline:
# identify the plant type, then route to the matching disease model.

ACTIVE_CROPS = [
    "Tomato",
    "Potato",
    "Corn_(maize)",
    "Pepper,_bell",
    "Apple",
    "Blueberry",
    "Cherry_(including_sour)",
    "Grape",
    "Orange",
    "Peach",
    "Raspberry",
    "Soybean",
    "Squash",
    "Strawberry",
]


# Training settings for the plant/crop classifier
@dataclass
class CropTrainConfig:
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 10
    lr: float = 1e-3
    seed: int = 42
    val_frac: float = 0.15

    # Cap the number of images per crop to keep training manageable
    max_per_crop_train: int = 800
    max_per_crop_val: int = 200


# Use MPS on Mac if available, otherwise CPU
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Clean up crop labels for display
def pretty_crop_name(name: str) -> str:
    return name.replace("_(maize)", "").replace(",", "").replace("_", " ")


# Build a crop → index mapping and make sure requested crops exist
def build_crop_mappings(root: str, active_crops: List[str]) -> Dict[str, int]:
    all_classes = list_all_classes(root)

    found_crops = sorted({crop_name_from_class(c) for c in all_classes})
    missing = [c for c in active_crops if c not in found_crops]
    if missing:
        raise RuntimeError(f"These crops were not found in dataset folders: {missing}")

    crop_to_idx = {crop: i for i, crop in enumerate(active_crops)}
    return crop_to_idx


# Build train/validation subsets for the crop classifier
def make_crop_subsets(root: str, cfg: CropTrainConfig, active_crops: List[str]):
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

    crop_to_idx = build_crop_mappings(root, active_crops)

    class_idx_to_crop = {
        idx: crop_name_from_class(class_name)
        for idx, class_name in enumerate(base_train.classes)
    }

    crop_to_sample_indices = {crop: [] for crop in active_crops}
    for i, (_, disease_class_idx) in enumerate(base_train.samples):
        crop = class_idx_to_crop[disease_class_idx]
        if crop in crop_to_sample_indices:
            crop_to_sample_indices[crop].append(i)

    train_indices = []
    val_indices = []

    for crop in active_crops:
        crop_indices = crop_to_sample_indices[crop]
        if not crop_indices:
            raise RuntimeError(f"No images found for crop: {crop}")

        tr, va = train_val_split(crop_indices, val_frac=cfg.val_frac, seed=cfg.seed)
        tr = tr[:cfg.max_per_crop_train]
        va = va[:cfg.max_per_crop_val]

        train_indices.extend(tr)
        val_indices.extend(va)

    train_ds = Subset(base_train, train_indices)
    val_ds = Subset(base_val, val_indices)

    return base_train, train_ds, val_ds, crop_to_idx, class_idx_to_crop


# Convert disease-class labels into crop-class labels
def remap_batch_to_crop_labels(
    y: torch.Tensor,
    class_idx_to_crop: Dict[int, str],
    crop_to_idx: Dict[str, int],
) -> torch.Tensor:
    out = torch.empty_like(y)
    for i in range(len(y)):
        disease_idx = int(y[i].item())
        crop_name = class_idx_to_crop[disease_idx]
        out[i] = crop_to_idx[crop_name]
    return out


# Train the crop classifier
def train_crop_classifier(out_path: str, cfg: CropTrainConfig, active_crops: List[str]) -> None:
    set_seed(cfg.seed)
    device = get_device()
    print("Device:", device)

    root = data_root_color()
    print("Dataset root:", root)
    print("Active crops:", active_crops)

    _, train_ds, val_ds, crop_to_idx, class_idx_to_crop = make_crop_subsets(root, cfg, active_crops)

    print("Num crop classes:", len(crop_to_idx))
    print("Train images:", len(train_ds))
    print("Val images:", len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, len(crop_to_idx))

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
            y_crop = remap_batch_to_crop_labels(y, class_idx_to_crop, crop_to_idx).to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y_crop)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y_crop).sum().item()
            seen += x.size(0)

        train_loss = total_loss / max(1, seen)
        train_acc = correct / max(1, seen)

        model.eval()
        v_correct, v_seen = 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y_crop = remap_batch_to_crop_labels(y, class_idx_to_crop, crop_to_idx).to(device)

                logits = model(x)
                v_correct += (logits.argmax(1) == y_crop).sum().item()
                v_seen += x.size(0)

        val_acc = v_correct / max(1, v_seen)

        print(
            f"Epoch {epoch+1}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "crop_to_idx": crop_to_idx,
            "idx_to_crop": {v: k for k, v in crop_to_idx.items()},
            "image_size": cfg.image_size,
            "active_crops": active_crops,
            "arch": "mobilenet_v2",
        },
        out_path,
    )
    print("Saved crop classifier to:", out_path)


# Load a saved crop classifier checkpoint
def load_crop_classifier(ckpt_path: str):
    device = get_device()
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Crop checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    num_classes = len(ckpt["crop_to_idx"])
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    idx_to_crop = (
        {int(k): v for k, v in ckpt["idx_to_crop"].items()}
        if any(isinstance(k, str) for k in ckpt["idx_to_crop"].keys())
        else ckpt["idx_to_crop"]
    )

    return model, device, ckpt["image_size"], idx_to_crop


# Predict crop type for one image
@torch.no_grad()
def predict_crop_image(ckpt_path: str, image_path: str):
    model, device, image_size, idx_to_crop = load_crop_classifier(ckpt_path)

    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    pred_idx = int(torch.argmax(probs).item())
    pred_crop = idx_to_crop[pred_idx]
    pred_prob = float(probs[pred_idx].item())

    all_results = [(idx_to_crop[i], float(probs[i].item())) for i in range(len(probs))]
    all_results = sorted(all_results, key=lambda x: x[1], reverse=True)

    return pred_crop, pred_prob, all_results


# Evaluate crop classifier on its validation split
def evaluate_crop_classifier(ckpt_path: str):
    model, device, _, idx_to_crop = load_crop_classifier(ckpt_path)
    active_crops = [idx_to_crop[i] for i in range(len(idx_to_crop))]

    cfg = CropTrainConfig()
    root = data_root_color()

    _, _, val_ds, crop_to_idx, class_idx_to_crop = make_crop_subsets(root, cfg, active_crops)
    loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu()

            y_crop = remap_batch_to_crop_labels(y, class_idx_to_crop, crop_to_idx)

            correct += (preds == y_crop).sum().item()
            total += len(y)

    acc = correct / max(total, 1)
    print(f"\nCrop classifier validation accuracy: {acc*100:.2f}%")
    print(f"Correct: {correct} / {total}")


# Ground-truth crop comes from the parent class folder
def ground_truth_crop_from_path(image_path: str) -> str:
    disease_folder = os.path.basename(os.path.dirname(image_path))
    return crop_name_from_class(disease_folder)


# Show image + probability bar chart for crop prediction
def visualize_crop_prediction(ckpt_path: str, image_path: str):
    import matplotlib.pyplot as plt

    gt_crop = ground_truth_crop_from_path(image_path)
    pred_crop, pred_prob, all_results = predict_crop_image(ckpt_path, image_path)
    img = Image.open(image_path).convert("RGB")

    gt_pretty = pretty_crop_name(gt_crop)
    pred_pretty = pretty_crop_name(pred_crop)

    sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)
    labels_plain = [pretty_crop_name(name) for name, _ in sorted_results]
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
    ax1.set_title("Class probabilities (Plant-type classifier)")
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

    print("Ground truth:", gt_crop)
    print(f"Prediction: {pred_crop} (confidence={pred_prob:.3f})")


# Full inference pipeline:
# 1) predict crop
# 2) load matching disease model
# 3) predict disease
def predict_full_pipeline(crop_ckpt_path: str, image_path: str):
    pred_crop, crop_conf, crop_results = predict_crop_image(crop_ckpt_path, image_path)

    disease_ckpt_path = default_ckpt_path_for_crop(pred_crop)

    if not os.path.isfile(disease_ckpt_path):
        return {
            "pred_crop": pred_crop,
            "crop_conf": crop_conf,
            "crop_results": crop_results,
            "disease_supported": False,
            "disease_label": None,
            "disease_conf": None,
            "plant_status": None,
            "disease_ckpt_path": disease_ckpt_path,
        }

    disease_label, disease_conf = predict_crop_disease(disease_ckpt_path, image_path)

    healthy = "healthy" in disease_label.lower()
    plant_status = "Healthy" if healthy else "Diseased"

    return {
        "pred_crop": pred_crop,
        "crop_conf": crop_conf,
        "crop_results": crop_results,
        "disease_supported": True,
        "disease_label": disease_label,
        "disease_conf": disease_conf,
        "plant_status": plant_status,
        "disease_ckpt_path": disease_ckpt_path,
    }


# Visual version of the full pipeline
def visualize_full_pipeline(crop_ckpt_path: str, image_path: str):
    print("\n=== CROP CLASSIFICATION ===")
    visualize_crop_prediction(crop_ckpt_path, image_path)

    result = predict_full_pipeline(crop_ckpt_path, image_path)

    print("\n=== PIPELINE RESULT ===")
    print(f"Predicted crop: {result['pred_crop']} (confidence={result['crop_conf']:.6f})")

    if result["disease_supported"]:
        print(f"Disease prediction: {result['disease_label']} (confidence={result['disease_conf']:.6f})")
        print(f"Plant status: {result['plant_status']}")

        print("\n=== DISEASE CLASSIFICATION ===")
        visualize_crop_disease_prediction(result["disease_ckpt_path"], image_path)
    else:
        print(f"No disease checkpoint found for crop: {result['pred_crop']}")
        print(f"Expected checkpoint path: {result['disease_ckpt_path']}")


# Helpful debugging function: sample a few images from each crop
def debug_sample_predictions(ckpt_path: str, samples_per_crop: int = 3):
    root = data_root_color()

    for crop in ACTIVE_CROPS:
        print(f"\n=== {crop} ===")
        shown = 0

        for class_name in sorted(os.listdir(root)):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            if crop_name_from_class(class_name) != crop:
                continue

            for fname in sorted(os.listdir(class_dir)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                image_path = os.path.join(class_dir, fname)
                pred_crop, pred_prob, all_results = predict_crop_image(ckpt_path, image_path)

                print(f"\nGT={crop} | Pred={pred_crop} | Conf={pred_prob:.6f} | File={fname}")
                for name, p in all_results:
                    print(f"  {name}: {p:.10f}")

                shown += 1
                if shown >= samples_per_crop:
                    break

            if shown >= samples_per_crop:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", type=str, default=None, help="Path to image")

    parser.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join("..", "artifacts", "crop_mobilenetv2.pt")
    )

    parser.add_argument("--debug_samples", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--viz", action="store_true", help="Show crop probability chart")
    parser.add_argument("--pipeline", action="store_true", help="Run crop classifier, then route to disease model")

    args = parser.parse_args()

    if args.train:
        cfg = CropTrainConfig()
        train_crop_classifier(args.ckpt, cfg, ACTIVE_CROPS)

    elif args.evaluate:
        evaluate_crop_classifier(args.ckpt)

    elif args.debug_samples:
        debug_sample_predictions(args.ckpt)

    elif args.predict:
        if args.pipeline:
            visualize_full_pipeline(args.ckpt, args.predict)
        elif args.viz:
            visualize_crop_prediction(args.ckpt, args.predict)
        else:
            result = predict_full_pipeline(args.ckpt, args.predict)
            print(f"Predicted crop: {result['pred_crop']} (confidence={result['crop_conf']:.6f})")
            if result["disease_supported"]:
                print(f"Disease prediction: {result['disease_label']} (confidence={result['disease_conf']:.6f})")
                print(f"Plant status: {result['plant_status']}")
            else:
                print("No disease model available for predicted crop.")
                print(f"Expected checkpoint path: {result['disease_ckpt_path']}")

    else:
        print("Use --train or --predict path/to/image.jpg or --evaluate")


if __name__ == "__main__":
    main()