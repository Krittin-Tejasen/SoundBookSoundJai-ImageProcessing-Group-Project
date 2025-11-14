from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import cv2, time, os
import json
from processing import remove_background, enhance_for_ocr_auto, pytesseract_ocr, text_to_speech
from ml_model import load_models, process_for_ocr
from gtts import gTTS 


base_dir = "C:/Users/User/OneDrive/Desktop/SoundBookSoundJai-ImageProcessing-Group-Project/models"
rf_model, cnn_model, device = load_models(base_dir)

# ----------------------------------------
# -------------- Files setup -------------
# ----------------------------------------

def load_seen_files():
    """Load processed file IDs from disk."""
    if os.path.exists("processed_files.json"):
        try:
            with open("processed_files.json", "r") as f:
                return set(json.load(f))
        except:
            return set()
    return set()


def save_seen_files(seen):
    """Save processed file IDs to disk."""
    with open("processed_files.json", "w") as f:
        json.dump(list(seen), f)

# ----------------------------------------
# ---------- Google Drive setup ----------
# ----------------------------------------

def connect_drive():
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("credentials.json")
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    gauth.SaveCredentialsFile("credentials.json")
    return GoogleDrive(gauth)

def get_folder_files(drive, folder_id):
    return drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()

def upload_file_to_drive(drive, file_path, folder_id):
    file_name = os.path.basename(file_path)
    gfile = drive.CreateFile({'title': file_name, 'parents': [{'id': folder_id}]})
    gfile.SetContentFile(file_path)
    gfile.Upload()
    print(f"[Drive] Uploaded: {file_name}")

# --------------------------------------
# ---------- Image Processing ----------
# --------------------------------------

def process_image_file(local_path, drive, output_folder_id, text_folder_id, audio_folder_id):
    print(f"[INFO] Processing {local_path}")

    # Ensure local folders exist

    os.makedirs("downloads/bg_removed", exist_ok=True)
    os.makedirs("downloads/processed", exist_ok=True)
    os.makedirs("downloads/text", exist_ok=True)
    os.makedirs("downloads/audio", exist_ok=True)

    filename = os.path.basename(local_path)
    base_name, _ = os.path.splitext(filename)
    bg_removed_path = os.path.join("downloads", "bg_removed", f"bg_removed_{base_name}.png")

    # ML quality check
    _ = process_for_ocr(local_path, rf_model, cnn_model, device)

    # ถ้าคุณภาพแย่ → ถามผู้ใช้ว่าจะ Continue หรือ Skip
    if _["is_bad"]:
        print("\n⚠️ Image Quality Warning:")
        print("   →", _["message"])
        choice = input("Do you want to continue? (c = continue, s = skip): ").strip().lower()

        if choice == "s":
            print("[INFO] User chose to skip this image.\n")
            return  # ยกเลิกการประมวลผลภาพนี้ทันที


    # Remove image background
    bg_removed_path = remove_background(local_path, bg_removed_path)
    print(f"[INFO] Background removed -> {bg_removed_path}")

    # Enhance image
    img = cv2.imread(bg_removed_path)
    if img is None:
        print("[WARN] Could not load background-removed image.")
        return
    
    processed = enhance_for_ocr_auto(img)
    processed_path = os.path.join("downloads", "processed", f"processed_{base_name}.png")
    cv2.imwrite(processed_path, processed)
    print(f"[DONE] Enhanced image saved -> {processed_path}")

    # OCR
    processed_img = cv2.imread(processed_path)
    if processed_img is not None:
        text = pytesseract_ocr(processed_img)
        text_file_path = os.path.join("downloads", "text", f"{base_name}.txt")
        with open(text_file_path, "w", encoding="utf-8") as f:
            f.write(text.strip())
        print(f"[OCR] Extracted text saved -> {text_file_path}")
    else:
        print("[WARN] Could not read processed image for OCR.")

    clean_text = text.replace("\n", " ").strip()  

    # Text to Speech
    audio_path = text_to_speech(clean_text, base_name)

    # Upload processed image back to Drive
    upload_file_to_drive(drive, processed_path, output_folder_id)
    upload_file_to_drive(drive, text_file_path, text_folder_id)
    if audio_path is not None:
        upload_file_to_drive(drive, audio_path, audio_folder_id)
    print(f"[UPLOAD] Processed image uploaded to Drive ✅")


# ----------------------------------------
# ------------- Main Watcher -------------
# ----------------------------------------
def watch_drive_folder(input_folder_id, output_folder_id, text_folder_id, audio_folder_id, poll_interval=10):
    
    drive = connect_drive()
    seen = load_seen_files()
    os.makedirs("downloads", exist_ok=True)

    print(f"👁 Watching Google Drive folder ID: {input_folder_id}")
    while True:
        try:
            files = get_folder_files(drive, input_folder_id)
            for f in files:
                #skip if processed before
                if f['id'] in seen:
                    continue

                if f['id'] not in seen and f['mimeType'].startswith('image/'):
                    print(f"[NEW] {f['title']}")
                    local_path = os.path.join("downloads", f['title'])
                    f.GetContentFile(local_path)
                    process_image_file(local_path, drive, output_folder_id, text_folder_id, audio_folder_id)
                    # mark as processed
                    seen.add(f['id'])
                    save_seen_files(seen)

            time.sleep(poll_interval)
        except Exception as e:
            print("[ERROR]", e)
            time.sleep(30)

if __name__ == "__main__":
    input_folder_id = "168KayTrlk3_r8ScA6qHmTZNfDMshkg6I"       #   Images folder
    output_folder_id = "1k2EvExrg5-jH3VGDinMZhuOXM8d2XlSn"      #   Removed_bg_images folder
    text_folder_id = "1IbW5zPDM9_0Bv5y296zuVlzqMNAblOjC"        #   Text folder
    audio_folder_id = "1fqrLE3KtwQVCRez4wwGMlSKdxEs3Px5v"       #   Audio folder
    watch_drive_folder(input_folder_id, output_folder_id, text_folder_id, audio_folder_id)
