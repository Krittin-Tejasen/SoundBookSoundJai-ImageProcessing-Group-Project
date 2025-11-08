from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import cv2, time, os
from processing import remove_background, enhance_for_ocr_auto
from PIL import Image


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

# --------------------------------------
# ---------- Image Processing ----------
# --------------------------------------

def process_image_file(local_path, drive, output_folder_id):
    print(f"[INFO] Processing {local_path}")

    # Ensure local folders exist
    os.makedirs("downloads/bg_removed", exist_ok=True)
    os.makedirs("downloads/processed", exist_ok=True)

    filename = os.path.basename(local_path)
    bg_removed_path = os.path.join("downloads", "bg_removed", f"bg_removed_{filename}")

    # Remove image background
    bg_removed_path = remove_background(local_path, bg_removed_path)
    print(f"[INFO] Background removed -> {bg_removed_path}")

    # Enhance image
    img = cv2.imread(bg_removed_path)
    if img is None:
        print("[WARN] Could not load background-removed image.")
        return
    
    processed = enhance_for_ocr_auto(img)
    processed_path = os.path.join("downloads", "processed", f"processed_{filename}")
    cv2.imwrite(processed_path, processed)
    print(f"[DONE] Enhanced image saved -> {processed_path}")

    # Upload processed image back to Drive
    upload_file_to_drive(drive, processed_path, output_folder_id)
    print(f"[UPLOAD] Processed image uploaded to Drive ✅")

def upload_file_to_drive(drive, file_path, folder_id):
    file_name = os.path.basename(file_path)
    gfile = drive.CreateFile({'title': file_name, 'parents': [{'id': folder_id}]})
    gfile.SetContentFile(file_path)
    gfile.Upload()
    print(f"[Drive] Uploaded: {file_name}")

# ----------------------------------------
# ------------- Main Watcher -------------
# ----------------------------------------
def watch_drive_folder(input_folder_id, output_folder_id, poll_interval=10):
    drive = connect_drive()
    seen = set()
    os.makedirs("downloads", exist_ok=True)

    print(f"👁 Watching Google Drive folder ID: {input_folder_id}")
    while True:
        try:
            files = get_folder_files(drive, input_folder_id)
            for f in files:
                if f['id'] not in seen and f['mimeType'].startswith('image/'):
                    print(f"[NEW] {f['title']}")
                    local_path = os.path.join("downloads", f['title'])
                    f.GetContentFile(local_path)
                    process_image_file(local_path, drive, output_folder_id)
                    seen.add(f['id'])
            time.sleep(poll_interval)
        except Exception as e:
            print("[ERROR]", e)
            time.sleep(30)

if __name__ == "__main__":
    input_folder_id = "1UTz5qx_RtAyL_IzzKstMpkw1w804kurY"          # 📥 Images folder
    output_folder_id = "1PUh2BB5taTWTMZVPW_qVOi62yW9NA-Xb"     # 📤 Removed_bg_images folder
    watch_drive_folder(input_folder_id, output_folder_id)
