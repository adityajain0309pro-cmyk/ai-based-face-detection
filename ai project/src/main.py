"""Webcam-based face recognition and simple GUI front end.

This module provides two main entry points:

- ``recognize_from_webcam``: raw OpenCV window that performs real-time
    recognition from the webcam.
- ``launch_gui``: a small Tkinter application with tabs that lets you
    register faces, run recognition, and inspect stored metadata.

The default behavior when running ``python -m src.main`` is to launch
the GUI so non-technical users can interact easily.
"""

from typing import Dict, List, Tuple, Optional

import json
import threading

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

from .config import FACES_DIR, CAMERA_INDEX, METADATA_FILE
from .register_face import register_person, _update_metadata


def _ensure_cv2_face_available() -> None:
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "cv2.face module not available. Make sure 'opencv-contrib-python' is installed."
        )

def _load_metadata() -> Dict[str, Dict[str, str]]:
    if METADATA_FILE.exists():
        return json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    return {}


def load_training_data() -> Tuple[List[np.ndarray], List[int], Dict[int, str]]:
    """Load grayscale face images and integer labels from the faces directory.

    Returns faces (image ROIs), labels (int IDs), and a mapping from
    label ID to the folder key under data/faces.
    """
    faces: List[np.ndarray] = []
    labels: List[int] = []
    label_map: Dict[int, str] = {}

    if not FACES_DIR.exists():
        return faces, labels, label_map

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    current_label = 0
    for person_dir in FACES_DIR.iterdir():
        if not person_dir.is_dir():
            continue

        label_map[current_label] = person_dir.name

        for img_path in person_dir.glob("*.jpg"):
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detections = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(detections) == 0:
                continue

            (x, y, w, h) = detections[0]
            roi_gray = gray[y : y + h, x : x + w]
            faces.append(roi_gray)
            labels.append(current_label)

        current_label += 1

    return faces, labels, label_map


def create_recognizer() -> Tuple[Optional["cv2.face_LBPHFaceRecognizer"], Dict[int, str]]:
    """Create and train the recognizer, returning it and a label->display map.

    The display map contains strings like "ID - Name" for overlay.
    If no valid training data is found, the recognizer will be None.
    """
    _ensure_cv2_face_available()

    faces, labels, label_map = load_training_data()

    metadata = _load_metadata()

    label_display: Dict[int, str] = {}
    for label_id, folder_key in label_map.items():
        info = metadata.get(folder_key)
        if info:
            student_id = info.get("student_id", "").strip()
            name = info.get("name", "").strip()
            if student_id and name:
                label_display[label_id] = f"{student_id} - {name}"
            elif name:
                label_display[label_id] = name
            else:
                label_display[label_id] = folder_key
        else:
            label_display[label_id] = folder_key

    if not faces:
        print("[WARN] No training data found. Register faces using register_face.py first.")
        # No training possible; return None so caller can skip prediction
        return None, label_display

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    return recognizer, label_display


def recognize_from_webcam() -> None:
    recognizer, label_display = create_recognizer()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    print("[INFO] Starting webcam. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in detections:
            roi_gray = gray[y : y + h, x : x + w]

            label_name = "Unknown"
            if recognizer is not None and hasattr(recognizer, "predict") and label_display:
                label_id, confidence = recognizer.predict(roi_gray)
                # Lower confidence is better; threshold is empirical.
                if confidence < 80 and label_id in label_display:
                    label_name = label_display[label_id]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y - 25), (x + w, y), (0, 255, 0), cv2.FILLED)
            cv2.putText(
                frame,
                label_name,
                (x + 4, y - 7),
                cv2.FONT_HERSHEY_DUPLEX,
                0.55,
                (0, 0, 0),
                1,
            )

        cv2.imshow("Face Recognition (OpenCV)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


class FaceApp(tk.Tk):
    """Simple Tkinter GUI with tabs for registration, recognition, and data."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Face Recognition Demo")
        self.geometry("700x400")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self._build_register_tab(notebook)
        self._build_recognize_tab(notebook)
        self._build_data_tab(notebook)

    # ---------- Register tab ----------

    def _build_register_tab(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Register Face")

        ttk.Label(frame, text="Student ID:").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        self.entry_student_id = ttk.Entry(frame, width=30)
        self.entry_student_id.grid(row=0, column=1, padx=8, pady=8, sticky="w")

        ttk.Label(frame, text="Full Name:").grid(row=1, column=0, sticky="w", padx=8, pady=8)
        self.entry_full_name = ttk.Entry(frame, width=30)
        self.entry_full_name.grid(row=1, column=1, padx=8, pady=8, sticky="w")

        ttk.Label(frame, text="Images to capture:").grid(row=2, column=0, sticky="w", padx=8, pady=8)
        self.entry_num_images = ttk.Entry(frame, width=10)
        self.entry_num_images.insert(0, "5")
        self.entry_num_images.grid(row=2, column=1, padx=8, pady=8, sticky="w")

        ttk.Button(frame, text="Start Capture", command=self._on_register_clicked).grid(
            row=3,
            column=0,
            columnspan=2,
            pady=16,
        )

        info = (
            "Instructions:\n"
            "- After clicking 'Start Capture', a webcam window opens.\n"
            "- Press SPACE in that window to capture an image.\n"
            "- Press 'q' in the webcam window to abort early."
        )
        ttk.Label(frame, text=info, justify="left").grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="w",
            padx=8,
            pady=8,
        )

    def _on_register_clicked(self) -> None:
        student_id = self.entry_student_id.get().strip()
        full_name = self.entry_full_name.get().strip()

        if not full_name and not student_id:
            messagebox.showerror("Missing data", "Please enter at least a Student ID or a Full Name.")
            return

        num_images_str = self.entry_num_images.get().strip() or "5"
        try:
            num_images = int(num_images_str)
            if num_images <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid number", "Images to capture must be a positive integer.")
            return

        safe_name = "_".join(full_name.split()) or "unknown"
        person_key = f"{student_id}_{safe_name}" if student_id else safe_name

        # Run the potentially long-running OpenCV capture in a thread
        def worker() -> None:
            try:
                saved = register_person(person_key, num_images=num_images)
                if saved > 0:
                    _update_metadata(person_key, student_id or person_key, full_name or person_key)
                    message = f"Registered {saved} images for {full_name or person_key}."
                    self._show_info_async("Registration complete", message)
                else:
                    self._show_info_async("No images saved", "No images were captured; metadata not updated.")
            except Exception as exc:  # noqa: BLE001
                self._show_error_async("Error during registration", str(exc))

        threading.Thread(target=worker, daemon=True).start()

    # ---------- Recognition tab ----------

    def _build_recognize_tab(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Recognize")

        description = (
            "Click 'Start Recognition' to open the webcam window.\n"
            "Press 'q' in the webcam window to stop."
        )

        ttk.Label(frame, text=description, justify="left").pack(anchor="w", padx=8, pady=8)

        ttk.Button(frame, text="Start Recognition", command=self._on_recognize_clicked).pack(
            pady=16
        )

    def _on_recognize_clicked(self) -> None:
        def worker() -> None:
            try:
                recognize_from_webcam()
            except Exception as exc:  # noqa: BLE001
                self._show_error_async("Recognition error", str(exc))

        threading.Thread(target=worker, daemon=True).start()

    # ---------- Data tab ----------

    def _build_data_tab(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Registered Data")

        columns = ("person_key", "student_id", "name")
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=12)
        tree.heading("person_key", text="Person Key")
        tree.heading("student_id", text="Student ID")
        tree.heading("name", text="Full Name")
        tree.column("person_key", width=220)
        tree.column("student_id", width=120)
        tree.column("name", width=220)
        tree.pack(fill="both", expand=True, padx=8, pady=8)

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        self.data_tree = tree

        ttk.Button(frame, text="Refresh", command=self._refresh_data).pack(pady=(0, 8))

        self._refresh_data()

    def _refresh_data(self) -> None:
        for row in self.data_tree.get_children():
            self.data_tree.delete(row)

        data = _load_metadata()
        for person_key, info in sorted(data.items()):
            student_id = info.get("student_id", "")
            name = info.get("name", "")
            self.data_tree.insert("", "end", values=(person_key, student_id, name))

    # ---------- Utility helpers ----------

    def _show_info_async(self, title: str, message: str) -> None:
        self.after(0, lambda: messagebox.showinfo(title, message))

    def _show_error_async(self, title: str, message: str) -> None:
        self.after(0, lambda: messagebox.showerror(title, message))


def launch_gui() -> None:
    app = FaceApp()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
