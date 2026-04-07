# AI-Based Face Recognition System

This is a simple Python project for webcam-based face recognition.
It allows you to register faces and then recognize them in real time
using OpenCV's built-in LBPH face recognizer (no external dlib required).

## Features
- Register faces from your webcam into `data/faces/<name>`
- Real-time face detection and recognition using your webcam
- Easily extendable for more advanced use cases

## Requirements
- Python 3.9+ (recommended)
- A working webcam

Python packages are listed in `requirements.txt` and include:
- OpenCV with extra contrib modules (`opencv-contrib-python`)
- `numpy`

## Setup
1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: GUI front end (recommended)

Start the tabbed GUI application:

```bash
python -m src.main
```

The GUI has three tabs:
- **Register Face** – enter *Student ID* and *Full Name*, choose how many images to capture, then click **Start Capture**. A webcam window opens; press **SPACE** to capture each image and **q** to stop.
- **Recognize** – click **Start Recognition** to open the webcam recognition window. Press **q** in that window to stop.
- **Registered Data** – view all people stored in `data/people.json`. Use **Refresh** to reload after new registrations.

> Note: The VS Code task **"Run face recognition"** also runs `python -m src.main`, which now launches this GUI.

### Option 2: Command-line scripts

You can still use the original scripts directly:

1. Register a new face:
   ```bash
   python -m src.register_face
   ```
2. Start real-time recognition only (no GUI):
   ```bash
   python -m src.main
   ```

## Project Structure
- `src/config.py` – basic configuration (paths, camera index, tolerance)
- `src/register_face.py` – script to capture and store face images
- `src/main.py` – real-time face recognition from webcam
- `data/faces/` – folders containing registered face images

You can extend this project by adding a database, GUI, or model training pipeline.
