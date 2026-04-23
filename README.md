# DiseaseDetector

## Overview

DiseaseDetector is an AI-powered system for automatic crop and plant disease identification using leaf images. The system implements a two-stage deep learning pipeline:

1. A **crop classifier** identifies the plant type given a leaf (e.g., Apple, Tomato, Potato)
2. A **crop-specific disease classifier** predicts the disease affecting that plant

This design allows the system to mimic real-world usage, where the crop type is not known in advance.

The models are trained using the **PlantVillage dataset** and achieve high accuracy using transfer learning with MobileNetV2.

---

## Installation

Clone the repository and install dependencies:

```
git clone https://github.com/GideonFulton/DiseaseDetector.git
cd DiseaseDetector
pip install -r requirements.txt
```

---

## Running the Full Pipeline

To test the model on one of the given sample leaf images:

```
python src/plant_classifier.py \
--predict "samples/Tomato___Early_blight/0034a551-9512-44e5-ba6c-827f85ecc688___RS_Erly.B 9432.JPG" \
--pipeline \
--ckpt artifacts/crop_mobilenetv2.pt
```

Switch samples/Tomato___Early_blight/0034a551-9512-44e5-ba6c-827f85ecc688___RS_Erly.B 9432.JPG with the image path of the image you want to test.


This will output:

* Predicted crop type (output image)
* Predicted disease (output image)
* Plant status (Healthy / Diseased)

---

## Running Partial Pipeline 

If you already know the plant type you can run the disease detection model directly:

```
python src/crop_disease_classifier.py \
  --crop Tomato \
  --predict "samples/Tomato___Early_blight/0034a551-9512-44e5-ba6c-827f85ecc688___RS_Erly.B 9432.JPG" \
  --viz
```

---

## Sample Images

The `samples/` folder contains example images for each crop-disease class. These are included for demonstration purposes and may come from the same dataset used during training.

---

## Dataset (for Training)

This project uses the **PlantVillage dataset**:

https://www.kaggle.com/datasets/emmarex/plantdisease

To train models yourself, download the dataset and place it in:

```
plantvillage dataset/color
```

---

## Training the Models

### 1. Train Crop (Plant) Classifier

```
python src/plant_classifier.py --train --ckpt artifacts/crop_mobilenetv2.pt
```

This trains a model that predicts plant type across all supported crops.

---

### 2. Train Disease Classifiers (All Crops)

```
python src/crop_disease_classifier.py --train_all
```

This trains one disease classifier per crop and saves checkpoints in `artifacts/`.

---

### 3. Train Disease Classifier for One Crop

```
python src/crop_disease_classifier.py --crop Tomato --train --ckpt artifacts/tomato_mobilenetv2.pt
```

---

## Evaluating Models

Requires **PlantVillage dataset**

### Crop Classifier

```
python src/plant_classifier.py --evaluate --ckpt artifacts/crop_mobilenetv2.pt
```

### Individual Disease Classifier (example)

```
python src/crop_disease_classifier.py --crop Apple --evaluate --ckpt artifacts/apple_mobilenetv2.pt
```

## Evaluating All Disease Models

To evaluate the performance of all crop-specific disease classifiers at once, you can run:

```bash
python src/evaluate_all_disease_models.py
```

This script will:

* Load each crop-specific disease model from the `artifacts/` folder
* Run evaluation on its validation dataset
* Print accuracy metrics for each crop

Example output:

```
================================================================================
Crop: Apple
Checkpoint: artifacts/apple_mobilenetv2.pt
Apple disease validation accuracy: 99.58%
Correct: 474 / 476
Status: DONE for Apple
```

If a checkpoint is missing, the script will display:

```
Status: MISSING CHECKPOINT
```

This is useful for quickly verifying that all models were trained correctly and are performing as expected.

---

## Notes

* This evaluates **validation accuracy**, not a separate test set
* The dataset must be available locally (see Dataset section above)
* All disease model checkpoints must exist in `artifacts/`

---

## Project Structure

```
DiseaseDetector/
  artifacts/        # Saved model checkpoints
  samples/          # Example test images
  src/              # Source code
  requirements.txt
  README.md
```

The "plantvillage dataset" folder should be placed in the DiseaseDetector directory when downloaded.

---

## Notes

* Pretrained model weights are included in `artifacts/` for immediate use
* Training requires the PlantVillage dataset
* The system is designed as a prototype and performs best on clean dataset images

Credit ChatGPT for help with README formatting