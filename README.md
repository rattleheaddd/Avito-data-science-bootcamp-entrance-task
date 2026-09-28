# Text Orientation Classification

Binary image classifier for detecting whether an OCR text crop is upright or rotated by 180°.

The model outputs

```text
p_180 = P(image is rotated by 180°)
```

The final solution uses a lightweight **MobileNetV3-Small** model together with synthetic OCR data and rotation-symmetric training.

---

## Overview

The task is to classify text crops into two orientations:

- `0` — upright text
- `1` — text rotated by 180°

The final training pipeline is:

```text
ImageNet pretrained MobileNetV3-Small
                ↓
          MJSynth training
                ↓
      SynthTIGER fine-tuning
                ↓
   symmetry-aware fine-tuning
                ↓
 rotation-consistency inference
```

The main idea is to explicitly use the symmetry of the task:

```text
rot180(rot180(x)) = x
```

For a model logit `f(x)`, the final prediction is computed as

```python
sym_logit = (f(x) - f(rot180(x))) / 2
p_180 = sigmoid(sym_logit)
```

This guarantees that the two orientations are treated consistently.

---

## Model

The backbone is:

```text
MobileNetV3-Small
```

initialized with ImageNet weights.

The original classification head is replaced with a single-logit binary classifier.

The model was chosen because it provides a good trade-off between:

- accuracy;
- inference speed;
- model size;
- suitability for small OCR crops.

---

## Image preprocessing

Images are resized while preserving aspect ratio and padded to:

```text
192 × 64
```

Padding uses an approximately ImageNet-neutral RGB value:

```python
(124, 116, 104)
```

Images are then normalized using standard ImageNet statistics:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

During training, moderate augmentations are applied:

- small affine transformations;
- translation;
- scale changes;
- color jitter.

Most blur, noise, background and perspective variation is already introduced by SynthTIGER.

---

## Training data

### MJSynth

MJSynth / Synth90k is used as the initial OCR orientation training dataset.

Each original text crop is randomly assigned one of two orientations:

```text
upright  -> label 0
rot180   -> label 1
```

---

### TextOCR

TextOCR was also tested as a real-image fine-tuning dataset.

However, original TextOCR crops are not guaranteed to be upright, which introduces noise into orientation labels.

The experiments are preserved in the repository, but the final model does **not** use the TextOCR checkpoint as its final starting point.

---

### SynthTIGER

A custom SynthTIGER configuration is used to generate approximately 300k OCR text images.

The generator includes:

- multiple fonts;
- textured and solid backgrounds;
- blur;
- JPEG degradation;
- image resampling;
- Gaussian noise;
- moderate perspective distortion;
- varying text colors and sizes.

Only samples containing at least two alphanumeric characters are used:

```python
def valid_text(text):
    return sum(c.isalnum() for c in text) >= 2
```

After filtering:

```text
297,076 valid images
```

A deterministic 95/5 train-validation split is used.

Because every generated SynthTIGER image is upright by construction, clean `0° / 180°` orientation labels can be generated during training.

---

## Symmetry-aware training

The final fine-tuning stage uses two forward passes for every training image.

For an input image `x`:

```python
z = model(x)

x_rot = torch.rot90(x, 2, dims=(-2, -1))
z_rot = model(x_rot)

sym_logits = (z - z_rot) / 2
loss = BCEWithLogitsLoss()(sym_logits, labels)
```

The same model and the same weights are used for both forward passes.

This directly optimizes the same rotation-consistent prediction that is later used during inference.

---

## Validation results

Validation is performed on held-out SynthTIGER images.

Each validation crop is evaluated in both orientations.

| Model | Normal Brier | Symmetry Brier | Symmetry Accuracy |
|---|---:|---:|---:|
| MJSynth | 0.127564 | 0.101165 | 0.857907 |
| TextOCR | 0.055457 | 0.048370 | 0.932808 |
| MJSynth → SynthTIGER | 0.020422 | 0.015730 | 0.979061 |
| TextOCR → SynthTIGER | 0.023954 | 0.020024 | 0.972329 |
| + symmetry-aware fine-tuning | 0.018536 | **0.013267** | **0.982663** |

The final model therefore achieved:

```text
Brier score: 0.013267
Accuracy:    0.982663
1 - Brier:   0.986733
```

on the synthetic validation set.

These values are intended for comparing experiments using the same validation distribution and should not be interpreted as estimates of hidden-test performance.

---

## Project structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── model.py
├── dataset.py
├── train.py
│
├── synthtiger_dataset.py
├── finetune_synthtiger.py
├── finetune_synthtiger_sym.py
├── eval_synthtiger.py
│
├── textocr_dataset.py
├── finetune_textocr.py
├── validate_textocr.py
│
├── compare_test_models_v2.py
├── calibrate_temperature.py
│
├── make_final_submission.py
├── check_final_submission.py
│
├── configs/
│   └── config_orientation.yaml
└── weights/
    └── best_mjsynth_synthtiger_sym_v2.pth
```

Datasets, intermediate checkpoints, virtual environments and generated submissions are excluded from Git. The final checkpoint is included in weights/best_mjsynth_synthtiger_sym_v2.pth.

---

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it.

Windows:

```powershell
.venv\Scripts\activate
```

For GPU training, install PyTorch and torchvision separately using the CUDA build appropriate for your system.

Tested environment:

```text
Python 3.x
PyTorch 2.14.0+cu132
torchvision 0.29.0+cu132
CUDA-enabled GPU
BF16 support used during training/inference
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

---

## Expected data layout

Example:

```text
data/
├── synthtiger_train/
│   ├── gt.txt
│   └── images/
│       ├── 0/
│       ├── 1/
│       └── ...
│
└── test/
    └── images/
```

SynthTIGER annotations have the form:

```text
images\0\0.jpg  CRABBE
images\0\1.jpg  The
images\0\2.jpg  TEAKS
```

---

## SynthTIGER generation

SynthTIGER is installed separately.

Repository:

```text
https://github.com/clovaai/synthtiger
```

The custom configuration used for this project is stored in:

```text
configs/config_orientation.yaml
```

Clone SynthTIGER into the synthtiger/ subdirectory of this project, install it following its README, and run the commands below from this project root. The configuration stays in this project's configs/ directory. Its resources/ paths are resolved from the cloned SynthTIGER root; ensure the referenced corpora, fonts, images and colormap are present there.

```bash
cd synthtiger
synthtiger -o ../data/synthtiger_train -c 300000 -w 6 -s 42 -v examples/synthtiger/template.py SynthTiger ../configs/config_orientation.yaml
cd ..
```

The exact number of workers should be adjusted according to available RAM.

---

## Standard SynthTIGER fine-tuning

Starting from the MJSynth checkpoint:

```bash
python finetune_synthtiger.py \
    --data data/synthtiger_train \
    --checkpoint best_mobilenetv3_orientation.pth \
    --output best_mjsynth_synthtiger.pth \
    --lr 5e-5 \
    --epochs 2 \
    --batch-size 256 \
    --workers 2
```

---

## Symmetry-aware fine-tuning

First symmetry-aware epoch:

```bash
python finetune_synthtiger_sym.py \
    --checkpoint best_mjsynth_synthtiger.pth \
    --output best_mjsynth_synthtiger_sym.pth \
    --epochs 1 \
    --lr 1e-5 \
    --batch-size 128 \
    --workers 2
```

A second low-learning-rate symmetry-aware epoch was used for the final model:

```bash
python finetune_synthtiger_sym.py \
    --checkpoint best_mjsynth_synthtiger_sym.pth \
    --output best_mjsynth_synthtiger_sym_v2.pth \
    --epochs 1 \
    --lr 5e-6 \
    --batch-size 128 \
    --workers 2
```

---

## Evaluation

To compare checkpoints using the same fixed SynthTIGER validation split:

```bash
python eval_synthtiger.py
```

The validation split uses:

```text
seed = 42
validation fraction = 0.05
```

---

## Inference

Final inference uses rotation consistency:

```python
with torch.no_grad():
    z0 = model(x)

    x180 = torch.rot90(
        x,
        k=2,
        dims=(-2, -1),
    )

    z180 = model(x180)

    sym_logit = (z0 - z180) / 2
    p_180 = torch.sigmoid(sym_logit)
```

No temperature scaling is applied:

```text
T = 1.0
```

Temperature scaling was tested separately but produced negligible improvement.

---

## Creating predictions

Place test images in:

```text
data/test/images/
```

and run:

```bash
python make_final_submission.py
```

The resulting CSV contains:

```text
image_id,p_180
```

Before use, the output can be verified with:

```bash
python check_final_submission.py
```

The checker verifies:

- row count;
- column names;
- missing values;
- duplicated IDs;
- probability range;
- correspondence with `sample_submission.csv`.

---

## Notes

Datasets and generated submission files are intentionally excluded from the repository.

The final checkpoint is included in:

```text
weights/best_mjsynth_synthtiger_sym_v2.pth
```

The project focuses on a compact and fast solution rather than large vision architectures or model ensembles.