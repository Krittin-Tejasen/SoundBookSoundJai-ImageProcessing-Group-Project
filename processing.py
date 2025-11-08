import cv2
import numpy as np
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
import os
from gtts import gTTS
from rembg import remove
from PIL import Image

# --- CONFIG ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------- Utility Functions ----------
def ensure_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def ensure_bgr(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

# ---------- Image Analysis ----------
def analyze_image(img):
    gray = ensure_gray(img)
    brightness = np.mean(gray)
    contrast = gray.std()
    info = {
        "brightness": brightness,
        "contrast": contrast,
        "is_dark": brightness < 80,
        "is_bright": brightness > 180,
        "low_contrast": contrast < 40,
        "is_noisy": np.var(gray) > 2000,
    }
    return info

# ---------- Enhancement Functions ----------
def fix_brightness(img, strength=1.0):
    gray = ensure_gray(img)
    mean_val = np.mean(gray)
    gamma = 1.0
    if mean_val < 100:
        gamma = 0.6 * strength
    elif mean_val > 160:
        gamma = 1.4 * strength
    corrected = np.power(gray / 255.0, gamma)
    return np.uint8(np.clip(corrected * 255, 0, 255))

def enhance_contrast(img, strength=1.0):
    gray = ensure_gray(img)
    clip = 2.0 + (strength * 1.0)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return clahe.apply(gray)

def reduce_noise(img, strength=1.0):
    gray = ensure_gray(img)
    k = 3 if strength < 1.5 else 5
    return cv2.medianBlur(gray, k)

def normalize_lighting(img):
    gray = ensure_gray(img)
    background = cv2.GaussianBlur(gray, (55, 55), 0)
    normalized = cv2.divide(gray, background, scale=255)
    return normalized

def binarize_adaptive(img):
    gray = ensure_gray(img)
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )

def deskew_text_based(img):
    gray = ensure_gray(img)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    if lines is None:
        return gray
    angles = []
    for line in lines:
        rho, theta = line[0]
        deg = (theta * 180 / np.pi) - 90
        if -60 < deg < 60:
            angles.append(deg)
    if not angles:
        return gray
    angle = np.median(angles)
    print(f"[INFO] Detected angle: {angle:.2f}°")
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=255)
    return rotated

def remove_background(image_path, output_path):
    input_image = Image.open(image_path)
    output_image = remove(input_image)

    if output_image == "RGBA":
        output_image = output_image.convert("RGB")

    base, _ = os.path.splitext(output_path)
    output_path = base + ".png"
    output_image.save(output_path)
    return output_path

# ---------- OCR & Enhancement Pipeline ----------
def enhance_for_ocr_auto(image):
    img = image.copy()
    info = analyze_image(img)
    print("[INFO] Image analysis:", info)
    processed = img.copy()

    dark_strength = max(0, min(1, (100 - info["brightness"]) / 100)) if info["is_dark"] else 0
    contrast_strength = max(0, min(1, (40 - info["contrast"]) / 40)) if info["low_contrast"] else 0
    noise_strength = max(0, min(1, (np.var(ensure_gray(img)) - 2000) / 3000)) if info["is_noisy"] else 0
    print(f"[INFO] Adaptive strength: dark={dark_strength:.2f}, contrast={contrast_strength:.2f}, noise={noise_strength:.2f}")

    if dark_strength > 0:
        print(f"[ACTION] Brightness correction (strength={dark_strength:.2f})")
        processed = fix_brightness(ensure_bgr(processed), 1.0 + dark_strength)
        processed = enhance_contrast(ensure_bgr(processed), 1.0)

    if noise_strength > 0:
        print(f"[ACTION] Noise reduction (strength={noise_strength:.2f})")
        processed = reduce_noise(ensure_bgr(processed), 1.0 + noise_strength)

    if contrast_strength > 0:
        print(f"[ACTION] Contrast enhancement (strength={contrast_strength:.2f})")
        processed = enhance_contrast(ensure_bgr(processed), 1.0 + contrast_strength)

    if not (info["is_dark"] or info["low_contrast"] or info["is_noisy"]):
        print("[ACTION] Normal image → normalize lighting")
        processed = normalize_lighting(ensure_bgr(processed))

    print("[ACTION] Adaptive binarization...")
    processed = binarize_adaptive(ensure_bgr(processed))
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
    print("[ACTION] Deskewing...")
    processed = deskew_text_based(ensure_bgr(processed))
    print("[DONE] Enhancement completed ✅")
    return processed

# ---------- OCR & Speech ----------
def pytesseract_ocr(image):
    pil_img = Image.fromarray(image)
    config = '--psm 3'
    text = pytesseract.image_to_string(pil_img, lang='eng', config=config)
    print("\n--- OCR Result ---\n", text)
    return text

def speak_text(text):
    if not text.strip():
        print("[INFO] No text to speak.")
        return
    tts = gTTS(text)
    tts.save("output.mp3")
    os.system("start output.mp3" if os.name == "nt" else "mpg123 output.mp3")

