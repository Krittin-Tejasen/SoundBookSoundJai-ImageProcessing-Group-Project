=========================
SoundBookSoundJai Project
=========================

------------------------------------------
REQUIREMENTS
------------------------------------------
Python 3.10 or newer

System dependencies:
- Tesseract OCR engine
- Internet connection for Google Drive API

Python libraries (installed automatically from requirements.txt): "pip install -r requirements.txt"
- pydrive2
- pytesseract
- opencv-python
- numpy
- Pillow
- rembg
- onnxruntime
- gTTS
- matplotlib
- torch, torchvision, torchaudio
- scikit-learn, joblib
- tqdm
- PyYAML
- requests

------------------------------------------
INSTALLATION STEPS
------------------------------------------

1️. Clone or download this project:
    git clone https://github.com/Krittin-Tejasen/SoundBookSoundJai-ImageProcessing-Group-Project.git
    cd SoundBookSoundJai_Project

2. Create a virtual environment (recommended):
    python -m venv venv

3. Activate the virtual environment:
    On Windows:
        venv\Scripts\activate
    On macOS/Linux:
        source venv/bin/activate

4. Install all dependencies:
    pip install -r requirements.txt

5. Install Tesseract OCR engine:

      Windows:
        Download and install from:
        https://github.com/UB-Mannheim/tesseract/wiki
        Then make sure to set the path inside `processing.py`:
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

      macOS:
        brew install tesseract

      Linux (Ubuntu/Debian):
        sudo apt install tesseract-ocr

------------------------------------------
GOOGLE DRIVE SETUP
------------------------------------------

1. Go to https://console.cloud.google.com/
2. Create a new project (example: DriveOCRProject)
3. Enable the Google Drive API.
4. Create OAuth client credentials (type: Desktop App).
5. Download the `client_secrets.json` file.
6. Place it in the same folder as `drive_ocr_watcher.py`.
7. When you run the program for the first time, a browser window will open.
   Log in with your Google account and approve access.

A new file `credentials.json` will be created automatically for future runs.

------------------------------------------
FOLDER STRUCTURE (Google Drive)
------------------------------------------

You should have a folder on Google Drive like this:

    Project_image/
    ├── Images/                <- Upload images here
    ├── Removed_bg_images/     <- Processed (background removed) images appear here
    ├── Text/                  <- OCR results (.txt)
    └── Audio/                 <- Text-to-speech output (.mp3)

------------------------------------------
RUNNING THE PROGRAM
------------------------------------------

1. Activate your virtual environment (if not already active):
    venv\Scripts\activate

2. Run the watcher script:
    python drive_ocr_watcher.py

3. Upload new images to your Google Drive "Images" folder.

4. The script will automatically:
    - Detect new uploads
    - Download the image
    - ML check page quality
    - Remove the background
    - Enhance the image
    - Extract text (OCR)
    - Generate speech (.mp3)
    - Upload results back to Drive

5. The console will show logs for each processed file.

------------------------------------------
DO NOT UPLOAD THESE FILES TO GITHUB
------------------------------------------
- client_secrets.json  (contains private API credentials)
- credentials.json     (contains personal access token)
- downloads/           (temporary local files)
- venv/                (virtual environment)
- processed_files.json (runtime log)

------------------------------------------
TIP: RESETTING THE WATCHER
------------------------------------------
If you want to reprocess all files, delete:
    processed_files.json
Then rerun the script.

------------------------------------------
CREDITS
------------------------------------------
Developed by: Krittin Tejasen 6613112, Ruaengsiri Nantavit 6613122
Project: SoundBookSoundJai
Purpose: Assistive reading tool for visually impaired users