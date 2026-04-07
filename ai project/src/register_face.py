"""Simple script to register a new face and details.

This script captures a few images of a person's face from the webcam
and stores them in the data/faces/<person_key> folder.
It also records basic details (ID, name) in a metadata file.
"""

import json
from pathlib import Path

import cv2

from .config import FACES_DIR, CAMERA_INDEX, METADATA_FILE


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_metadata() -> dict:
    if METADATA_FILE.exists():
        return json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    return {}


def _save_metadata(data: dict) -> None:
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _update_metadata(person_key: str, student_id: str, full_name: str) -> None:
    data = _load_metadata()
    data[person_key] = {"student_id": student_id, "name": full_name}
    _save_metadata(data)


def register_person(person_key: str, num_images: int = 5) -> int:
    if not person_key:
        raise ValueError("person_key must not be empty")

    person_dir = FACES_DIR / person_key
    ensure_dir(person_dir)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    print(f"[INFO] Starting capture for '{person_key}'. Press 'q' to abort.")

    saved = 0
    while saved < num_images:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from camera.")
            break

        cv2.imshow("Register Face - Press space to capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            img_path = person_dir / f"{person_key}_{saved+1}.jpg"
            cv2.imwrite(str(img_path), frame)
            saved += 1
            print(f"[INFO] Saved image {saved}/{num_images}: {img_path}")
        elif key == ord("q"):
            print("[INFO] Aborted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Capture finished.")

    return saved


if __name__ == "__main__":
    student_id = input("Enter ID (e.g. roll/employee no.): ").strip()
    full_name = input("Enter full name: ").strip()

    # Use a stable key for folder + metadata
    safe_name = "_".join(full_name.split()) or "unknown"
    person_key = f"{student_id}_{safe_name}" if student_id else safe_name

    num_saved = register_person(person_key)
    if num_saved > 0:
        _update_metadata(person_key, student_id or person_key, full_name or person_key)
        print(f"[INFO] Registered {num_saved} images for {full_name or person_key} (ID: {student_id or person_key}).")
    else:
        print("[WARN] No images were saved; metadata not updated.")
