import os
import subprocess
import sys

CROPS = [
    "Apple",
    "Blueberry",
    "Cherry_(including_sour)",
    "Corn_(maize)",
    "Grape",
    "Orange",
    "Peach",
    "Pepper,_bell",
    "Potato",
    "Raspberry",
    "Soybean",
    "Squash",
    "Strawberry",
    "Tomato",
]


def safe_crop_name(crop_name: str) -> str:
    return (
        crop_name.lower()
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace(" ", "_")
    )


def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    project_root_dir = project_root()

    script_path = os.path.join(project_root_dir, "src", "crop_disease_classifier.py")
    artifacts_dir = os.path.join(project_root_dir, "artifacts")

    print("Evaluating all crop-specific disease models...\n")

    for crop in CROPS:
        ckpt_path = os.path.join(artifacts_dir, f"{safe_crop_name(crop)}_mobilenetv2.pt")

        print("=" * 80)
        print(f"Crop: {crop}")
        print(f"Checkpoint: {ckpt_path}")

        if not os.path.isfile(ckpt_path):
            print("Status: MISSING CHECKPOINT\n")
            continue

        cmd = [
            sys.executable,
            script_path,
            "--crop", crop,
            "--evaluate",
            "--ckpt", ckpt_path,
        ]

        result = subprocess.run(cmd, cwd=project_root_dir)

        if result.returncode != 0:
            print(f"Status: FAILED for {crop}\n")
        else:
            print(f"Status: DONE for {crop}\n")


if __name__ == "__main__":
    main()