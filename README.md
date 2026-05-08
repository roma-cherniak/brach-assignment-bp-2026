MY Simple ASL project with OpenCV 


Installation - pip install assignment_bp_roma_cherniak_2026



## Quick Start

### Predict from image file
```python
from asl_classifier import load_model, predict
from PIL import Image

model = load_model()
image = Image.open("hand.jpg")
label, confidence = predict(model, image)
print(f"Prediction: {label} ({confidence}%)")
```

### Try it live with your webcam
```python
from asl_classifier import load_model, run_webcam

model = load_model()
run_webcam(model)  # press Q to quit
```

## API Reference

### `load_model(weights_path=None)`
Loads the ASL classifier model with pretrained weights.


### `predict(model, image)`
Runs inference on a single PIL image.
- `model` — loaded model from `load_model()`
- `image` — PIL Image object

### `run_webcam(model)`
Opens webcam and runs live prediction in real time.
- `model` — loaded model from `load_model()`
- Press **Q** to quit

### `Net`
PyTorch `nn.Module` class implementing a 4-layer CNN for ASL classification.
- Input: RGB image of any size (resized to 128x128 internally)
- Output: class probabilities for 5 signs

## Evaluation Proposal

My primary metrics for evaluating the model's performance were validation accuracy during training, as well as the model's real-world ability to recognize these signs in practice.
While the results demonstrated promising progress, the model is far from optimal and requires further refinement and improvement.