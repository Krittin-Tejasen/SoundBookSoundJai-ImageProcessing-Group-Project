from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import cv2, time, os
import json
from processing import remove_background, enhance_for_ocr_auto, pytesseract_ocr, text_to_speech
from ml_model import load_models, process_for_ocr
from gtts import gTTS 
from playsound import playsound
import threading
import json
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


base_dir = "C:/Users/User/OneDrive/Desktop/SoundBookSoundJai_Project/models"
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

    # ML quality check: returns image and bad_quality flag
    img_read, bad_quality = process_for_ocr(local_path, rf_model, cnn_model, device)

    # If ML suspects bad quality, ask user to continue or skip
    if bad_quality:
        while True:
            resp = input(f"[DECISION] {filename} appears to have embedded images or poor quality. Continue processing? (y = continue / s = skip): ").strip().lower()
            if resp in ("y", "yes"):
                print("[DECISION] User chose to continue processing.")
                break
            elif resp in ("s", "skip", "n", "no"):
                print("[DECISION] User chose to skip this file.")
                return {"status": "skipped", "audio_path": None}
            else:
                print("Please enter 'y' to continue or 's' to skip.")

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
    text = ""
    if processed_img is not None:
        text = pytesseract_ocr(processed_img)
        text_file_path = os.path.join("downloads", "text", f"{base_name}.txt")
        with open(text_file_path, "w", encoding="utf-8") as f:
            f.write(text.strip())
        print(f"[OCR] Extracted text saved -> {text_file_path}")
    else:
        print("[WARN] Could not read processed image for OCR.")
        text_file_path = os.path.join("downloads", "text", f"{base_name}.txt")
        with open(text_file_path, "w", encoding="utf-8") as f:
            f.write("")

    clean_text = text.replace("\n", " ").strip()

    # Text to Speech
    audio_path = text_to_speech(clean_text, base_name)

    # Upload processed image back to Drive
    upload_file_to_drive(drive, processed_path, output_folder_id)
    upload_file_to_drive(drive, text_file_path, text_folder_id)
    if audio_path is not None:
        upload_file_to_drive(drive, audio_path, audio_folder_id)
    print(f"[UPLOAD] Processed image uploaded to Drive ✅")
    print(f"----------------------------------------------")

    return {"status": "processed", "audio_path": audio_path}


# ----------------------------------------
# ----------- Config & Audio -------------
# ----------------------------------------

# play_mode: "immediate", "batch", or "prompt"
PLAY_MODE = "prompt"
# For prompt mode, auto-decide after timeout_seconds if user doesn't reply
PROMPT_TIMEOUT_SECONDS = 100
# Use 'playsound' or 'pydub' backend. If playsound not installed, pip install playsound
PLAYBACK_BACKEND = "playsound"

def play_audio_blocking(path):
    try:
        playsound(path)
    except Exception as e:
        print("[AUDIO PLAY ERROR]", e)

def play_audio_nonblocking(path):
    t = threading.Thread(target=play_audio_blocking, args=(path,), daemon=True)
    t.start()
    return t

def save_audio_queue(paths, queue_file="audio_queue.json"):
    existing = []
    if os.path.exists(queue_file):
        try:
            with open(queue_file, "r", encoding="utf-8") as fq:
                existing = json.load(fq)
        except:
            existing = []
    existing.extend(paths)
    with open(queue_file, "w", encoding="utf-8") as fq:
        json.dump(existing, fq, ensure_ascii=False)

def load_audio_queue(queue_file="audio_queue.json"):
    if not os.path.exists(queue_file):
        return []
    try:
        with open(queue_file, "r", encoding="utf-8") as fq:
            return json.load(fq)
    except:
        return []

def play_queued_audios(queue_file="audio_queue.json"):
    q = load_audio_queue(queue_file)
    if not q:
        return
    print(f"[AUDIO] Playing queued {len(q)} audios")
    for p in q:
        play_audio_blocking(p)
    # clear queue
    with open(queue_file, "w", encoding="utf-8") as f:
        json.dump([], f)


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
            # collect new image files in this poll
            new_image_files = []
            for f in files:
                if f['id'] in seen:
                    continue
                if f['mimeType'].startswith('image/'):
                    new_image_files.append(f)

            if not new_image_files:
                time.sleep(poll_interval)
                continue

            # process each new file and gather results
            results = []
            for f in new_image_files:
                print(f"[NEW] {f['title']}")
                local_path = os.path.join("downloads", f['title'])
                f.GetContentFile(local_path)
                res = process_image_file(local_path, drive, output_folder_id, text_folder_id, audio_folder_id)
                # ensure res is a dict; convert None to error dict if needed
                if res is None:
                    res = {"status": "error", "audio_path": None}
                results.append((f, res))
                # mark as processed regardless of status so we don't re-download repeatedly
                seen.add(f['id'])
                save_seen_files(seen)

            # gather produced audio paths
            audio_paths = [r[1].get("audio_path") for r in results if r[1].get("audio_path")]
            if not audio_paths:
                time.sleep(poll_interval)
                continue

            # Decide playback based on PLAY_MODE
            if PLAY_MODE == "immediate":
                for ap in audio_paths:
                    print(f"[AUDIO] Playing {ap}")
                    play_audio_blocking(ap)

            elif PLAY_MODE == "batch":
                print(f"[AUDIO] {len(audio_paths)} files queued. Playing all now.")
                for ap in audio_paths:
                    play_audio_blocking(ap)

            elif PLAY_MODE == "prompt":
                if len(audio_paths) == 1:
                    print(f"[AUDIO] Playing single audio {audio_paths[0]}")
                    play_audio_blocking(audio_paths[0])
                else:
                    # prompt user with timeout; simple blocking prompt
                    print(f"[DECISION] {len(audio_paths)} audios are ready. Play now? (y = play now / n = skip / b = queue for later) ")
                    start = time.time()
                    answer = ""
                    try:
                        # user input (blocking) — will wait until user types or your environment times out
                        answer = input().strip().lower()
                    except Exception:
                        answer = ""
                    # timeout fallback: if empty and time exceeded, auto-skip
                    if not answer and (time.time() - start) > PROMPT_TIMEOUT_SECONDS:
                        answer = "n"
                        print()
                    if answer in ("y", "yes"):
                        for ap in audio_paths:
                            play_audio_blocking(ap)
                    elif answer in ("b", "batch"):
                        save_audio_queue(audio_paths)
                        print("[AUDIO] Saved to queue for later playback.")
                    else:
                        print("[AUDIO] Skipped playback.")
            prompt_play_queue()
            time.sleep(poll_interval)
        except Exception as e:
            print("[ERROR]", e)
            time.sleep(30)

def prompt_play_queue(queue_file="audio_queue.json"):
    """
    If there is a queue in queue_file, ask the user whether to play it now.
    If the user answers y/yes -> call play_queued_audios() and then continue starting the watcher.
    If the user answers anything else or the file is empty -> skip.
    """
    q = load_audio_queue(queue_file)
    if not q:
        return
    try:
        resp = input(f"[AUDIO QUEUE] Found {len(q)} files in '{queue_file}'. Play the queue now? (y = play / n = skip): ").strip().lower()
    except Exception:
        resp = ""
    if resp in ("y", "yes"):
        print("[AUDIO] Starting queued playback...")
        play_queued_audios(queue_file)
    else:
        print("[AUDIO] Skipping queued playback.")


if __name__ == "__main__":
    input_folder_id = "1UTz5qx_RtAyL_IzzKstMpkw1w804kurY"       #   Images folder
    output_folder_id = "1PUh2BB5taTWTMZVPW_qVOi62yW9NA-Xb"      #   Removed_bg_images folder
    text_folder_id = "1SHiMXazxwmiRFJ7PIOhQMS8PAJBhuHZV"        #   Text folder
    audio_folder_id = "1Ys4Dj1ivHEBWQNhUOzkKPoFHg9jZQuee"       #   Audio folder
    watch_drive_folder(input_folder_id, output_folder_id, text_folder_id, audio_folder_id)
