import os, cv2, joblib, torch
import numpy as np
from torchvision import transforms, models
from PIL import Image

def load_models(base_dir):
    rf_model = joblib.load(os.path.join(base_dir, "picture_detection_RF.pkl"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn_model = models.efficientnet_b0(weights=None)
    cnn_model.classifier[1] = torch.nn.Linear(cnn_model.classifier[1].in_features, 2)
    latest_model = os.path.join(base_dir, "cnn_fold5.pth")
    cnn_model.load_state_dict(torch.load(latest_model, map_location=device))
    cnn_model.to(device)
    cnn_model.eval()
    return rf_model, cnn_model, device

def extract_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    brightness = img.mean()
    contrast = img.std()
    edges = cv2.Canny(img, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    edge_var = np.var(edges)
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    hist = hist / hist.sum()
    entropy = -np.sum([p*np.log2(p) for p in hist if p!=0])
    return [brightness, contrast, edge_density, edge_var, entropy]

def predict_image(img_path, rf_model, cnn_model, device):
    features = np.array(extract_features(img_path)).reshape(1,-1)
    rf_pred = int(rf_model.predict(features)[0])
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = cnn_model(tensor)
        cnn_pred = int(torch.argmax(output,1).item())
    return rf_pred, cnn_pred

def process_for_ocr(img_path, rf_model, cnn_model, device):
    rf_pred, cnn_pred = predict_image(img_path, rf_model, cnn_model, device)
    if cnn_pred == 1 or rf_pred == 1:
        print("⚠️ Detected embedded images or poor quality page.")
    else:
        print("✅ Page appears clean and ready for OCR.")
    return cv2.imread(img_path),bad_quality

