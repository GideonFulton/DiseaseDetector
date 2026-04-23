import os
import random
from typing import List, Dict, Tuple

IMG_EXTS = (".jpg", ".jpeg", ".png")

def project_root() -> str:
    # src/ -> project root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def data_root_color() -> str:
    return os.path.join(project_root(), "plantvillage dataset", "color")

def list_tomato_classes(root_dir: str) -> List[str]:
    classes = []
    for d in os.listdir(root_dir):
        p = os.path.join(root_dir, d)
        if os.path.isdir(p) and d.startswith("Tomato___"):
            classes.append(d)
    return sorted(classes)

def count_images_per_class(root_dir: str, classes: List[str]) -> Dict[str, int]:
    out = {}
    for c in classes:
        p = os.path.join(root_dir, c)
        out[c] = sum(1 for f in os.listdir(p) if f.lower().endswith(IMG_EXTS))
    return out

def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def train_val_split(indices: List[int], val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    idx = indices[:]
    rng.shuffle(idx)
    split = int((1.0 - val_frac) * len(idx))
    return idx[:split], idx[split:]

def list_all_classes(root_dir: str) -> List[str]:
    classes = []
    for d in os.listdir(root_dir):
        p = os.path.join(root_dir, d)
        if os.path.isdir(p):
            classes.append(d)
    return sorted(classes)

def crop_name_from_class(class_name: str) -> str:
    # Examples:
    # "Tomato___Early_blight" -> "Tomato"
    # "Corn_(maize)___healthy" -> "Corn_(maize)"
    # "Pepper,_bell___healthy" -> "Pepper,_bell"
    if "___" in class_name:
        return class_name.split("___", 1)[0]
    return class_name

def list_crop_types(root_dir: str) -> List[str]:
    all_classes = list_all_classes(root_dir)
    crops = sorted({crop_name_from_class(c) for c in all_classes})
    return crops