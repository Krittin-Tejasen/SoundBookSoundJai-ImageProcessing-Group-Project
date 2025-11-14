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


def deskew_and_expand(image, angle, expand_threshold=10):
    """
    Deskew image and auto-expand canvas only if angle exceeds threshold.
    """
    (h, w) = image.shape[:2]
    # หาถ้าเอียงจริงเกิน threshold ค่อยขยาย
    if abs(angle) < expand_threshold:
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        border_value = 255 if len(image.shape) == 2 else (255,255,255)
        rotated = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=border_value)
    else:
        angle_rad = np.deg2rad(angle)
        cos = abs(np.cos(angle_rad))
        sin = abs(np.sin(angle_rad))
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        M[0, 2] += (new_w - w) // 2
        M[1, 2] += (new_h - h) // 2
        border_value = 255 if len(image.shape) == 2 else (255,255,255)
        rotated = cv2.warpAffine(image, M, (new_w, new_h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=border_value)
    return rotated

def detect_skew_angle_projection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)==3 else image.copy()
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    angles = np.arange(-45, 46, 1)  # เทสต์ทุกมุม -45 ถึง 45 องศา
    scores = []
    for angle in angles:
        M = cv2.getRotationMatrix2D((thresh.shape[1]//2, thresh.shape[0]//2), angle, 1)
        rotated = cv2.warpAffine(thresh, M, (thresh.shape[1], thresh.shape[0]),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
        proj = np.sum(rotated, axis=1)
        score = np.std(proj)
        scores.append(score)
    best_angle = angles[np.argmax(scores)]
    return best_angle

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

    # ---------- Adaptive Strength Calculation ----------
    dark_strength = max(0, min(1, (100 - info["brightness"]) / 100)) if info["is_dark"] else 0
    contrast_strength = max(0, min(1, (40 - info["contrast"]) / 40)) if info["low_contrast"] else 0
    noise_strength = max(0, min(1, (np.var(ensure_gray(img)) - 2000) / 3000)) if info["is_noisy"] else 0
    print(f"[INFO] Adaptive strength: dark={dark_strength:.2f}, contrast={contrast_strength:.2f}, noise={noise_strength:.2f}")

    # ---------- Apply Enhancements ----------
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
        print("[ACTION] Normal image -> normalize lighting")
        processed = normalize_lighting(ensure_bgr(processed))

    print("[ACTION] Adaptive binarization...")
    processed = binarize_adaptive(ensure_bgr(processed))

    # Morphology cleanup
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)

    print("[ACTION] Deskewing...")
    # แก้การ deskew: ตรวจหามุมเอียงและใช้ deskew_and_expand เพื่อป้องกันตกขอบ
    detected_angle = detect_skew_angle_projection(ensure_bgr(processed))  # ฟังก์ชันนี้ควร return มุมเอียง
    print(f"[INFO] Detected (text-based) angle: {detected_angle:.2f}°")
    processed = deskew_and_expand(ensure_bgr(processed), detected_angle)

    print("[DONE] Enhancement completed ✅")
    return processed

# ---------- OCR & Speech ----------
def pytesseract_ocr(image):
    pil_img = Image.fromarray(image)
    config = '--psm 3'
    text = pytesseract.image_to_string(pil_img, lang='eng', config=config)
    # print("\n--- OCR Result ---\n", text)
    return text

def text_to_speech(text: str, base_name:str):
    """Convert text to an MP3 file and return its local path."""
    if not text.strip():
        print("[INFO] No text to speak.")
        return None
    os.makedirs("downloads/audio", exist_ok=True)
    audio_path = os.path.join("downloads", "audio", f"{base_name}.mp3")
    tts = gTTS(text)
    tts.save(audio_path)
    print(f"[TTS] Audio file saved -> {audio_path}")
    return audio_path


def speak_text(text):
    if not text.strip():
        print("[INFO] No text to speak.")
        return
    tts = gTTS(text)
    tts.save("output.mp3")
    os.system("start output.mp3" if os.name == "nt" else "mpg123 output.mp3")

