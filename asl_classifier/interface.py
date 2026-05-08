import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from .model import Net

CLASSES = ['A', 'B', 'C', 'D', 'nothing']

inference_transform = transforms.Compose([
    transforms.Resize((148, 148)),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    weights_path = Path(__file__).parent / "model.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def predict(model, image):
    device = next(model.parameters()).device
    tensor = inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(1).item()
        confidence = torch.softmax(output, dim=1)[0][pred].item()

    return CLASSES[pred], round(confidence * 100, 2)



def run_webcam(model, device=None):
    import cv2
    from PIL import Image
    import torch
    import torchvision.transforms as transforms

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CLASSES = ['A', 'B', 'C', 'D', 'nothing']

    inference_transform = transforms.Compose([
        transforms.Resize((148, 148)),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    model.eval()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        x1, y1, x2, y2 = 100, 100, 400, 400
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Put hand here", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        crop = frame[y1:y2, x1:x2]
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        label, confidence = predict(model, img)

        color = (0, 255, 0) if confidence > 70 else (0, 0, 255)
        cv2.putText(frame, f"{label} ({confidence:.1f}%)", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        cv2.imshow("ASL Classifier", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
