MY Simple ASL project with OpenCV 


Installation - pip install assignment_bp_roma_cherniak_2026



## Quick Start

### Predict from image file
```python
from asl_classifier import load_model, predict
from PIL import Image

model = load_model()
image = Image.open("Your_hand_image.jpg")
label, confidence = predict(model, image)
print(f"Prediction: {label} ({confidence}%)")
```
hand image you can find in asl_alphabet_test folder 

### Try it live with your webcam
```python
from asl_classifier import load_model, run_webcam

model = load_model()
run_webcam(model)  # press Q to quit
```

## API Reference

### `load_model()`
Loads the ASL classifier model with pretrained weights.


### `predict(model, image)`
Runs inference on a single PIL image.
- `model` — loaded model from `load_model()`
- `image` — PIL Image object

### `run_webcam(model)`
Opens webcam and runs live prediction in real time.
- `model` — loaded model from `load_model()`
- Press **Q** to quit

## Evaluation Proposal

The model is trained on the [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet), a collection of 87,000 200×200 RGB images across 29 classes. For this project a 5-class subset (A, B, C, D, nothing) is used, split 80/10/10 into train/validation/test sets with stratified sampling to preserve class balance (~3,500 images per class in training).

**Metrics**
- Per-epoch validation accuracy (primary signal for early stopping)
- Per-class precision, recall, and F1-score on the held-out test set
- Confusion matrix to identify which sign pairs are most often confused

**Pipeline**
- Training: Adam optimizer, cross-entropy loss, 20 epochs, early stopping on validation loss plateau
- Preprocessing: `Resize((128, 128))`, per-channel mean/std normalization (0.5/0.5)
- Evaluation: the test set is never seen during training or hyperparameter tuning; final metrics are reported once against this set only
- Real-world sanity check: qualitative webcam testing across different lighting conditions and hand positions
