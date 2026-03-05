import cv2
import face_recognition
import numpy as np
import json
import os
import pickle
from datetime import datetime

KNOWN_FACES_DIR = "known_faces"
DB_FILE = os.path.join(KNOWN_FACES_DIR, "faces.pkl")
TOLERANCE = 0.6
FRAME_SCALE = 0.5  # Process at half resolution for speed


def load_database():
    """Load known face encodings and names from disk."""
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "names": []}


def save_database(db):
    """Save known face encodings and names to disk."""
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)


def save_face_image(name, frame, face_location):
    """Save a cropped face image for reference."""
    top, right, bottom, left = face_location
    face_img = frame[top:bottom, left:right]
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(person_dir, f"{timestamp}.jpg")
    cv2.imwrite(filepath, face_img)


def show_overlay(frame, lines, window="Face Recognition"):
    """Draw a dark overlay with text lines centered on the frame."""
    display = frame.copy()
    overlay = display.copy()
    h, w = display.shape[:2]
    box_h = 40 * len(lines) + 40
    y_start = h // 2 - box_h // 2
    cv2.rectangle(overlay, (0, y_start), (w, y_start + box_h), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
    for i, (text, color, scale) in enumerate(lines):
        cv2.putText(display, text, (20, y_start + 35 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
    cv2.imshow(window, display)


def get_name_from_gui(frame, existing_match=None):
    """Show an overlay on the video frame to type a name using keyboard.
    If existing_match is set, show it and let user confirm with ENTER or type a new name."""
    name = ""
    while True:
        lines = []
        if existing_match and not name:
            match_name, match_conf = existing_match
            lines.append((f"Already known as: {match_name} ({match_conf:.0f}%)", (0, 255, 0), 0.7))
            lines.append(("ENTER=add sample  Type new name to override  ESC=cancel", (200, 200, 200), 0.5))
            lines.append((name + "_", (0, 255, 255), 1.0))
        else:
            lines.append(("Type name and press ENTER (ESC to cancel):", (255, 255, 255), 0.7))
            lines.append((name + "_", (0, 255, 255), 1.0))
        show_overlay(frame, lines)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            return None
        elif key == 13 or key == 10:  # Enter
            if name.strip():
                return name.strip()
            elif existing_match:
                return existing_match[0]  # Confirm existing name
            return None
        elif key == 8 or key == 127:  # Backspace
            name = name[:-1]
        elif 32 <= key <= 126:  # Printable ASCII
            name += chr(key)


def main():
    db = load_database()
    print(f"Loaded {len(db['names'])} known face(s): {set(db['names'])}")
    print()
    print("Controls:")
    print("  L     - Learn/label the currently highlighted face")
    print("  TAB   - Cycle selection between detected faces")
    print("  Q/ESC - Quit")
    print()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    selected_face_idx = 0
    face_locations = []
    face_encodings = []
    face_names = []
    process_frame = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if process_frame:
            small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small)
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            face_locations = [
                (int(t / FRAME_SCALE), int(r / FRAME_SCALE),
                 int(b / FRAME_SCALE), int(l / FRAME_SCALE))
                for t, r, b, l in face_locations
            ]

            face_names = []
            face_confidences = []
            for encoding in face_encodings:
                name = "Unknown"
                confidence = 0.0
                if db["encodings"]:
                    distances = face_recognition.face_distance(db["encodings"], encoding)
                    best_idx = np.argmin(distances)
                    if distances[best_idx] < TOLERANCE:
                        name = db["names"][best_idx]
                        confidence = max(0.0, 1.0 - distances[best_idx]) * 100
                face_names.append(name)
                face_confidences.append(confidence)

            if selected_face_idx >= len(face_locations):
                selected_face_idx = 0

        process_frame = not process_frame

        for i, ((top, right, bottom, left), name, conf) in enumerate(zip(face_locations, face_names, face_confidences)):
            is_selected = (i == selected_face_idx)
            color = (0, 255, 255) if is_selected else (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            thickness = 3 if is_selected else 2

            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
            if name != "Unknown":
                label = f"{name} {conf:.0f}%"
            else:
                label = "Unknown"
            if is_selected:
                label += " [sel]"
            cv2.rectangle(frame, (left, bottom), (right, bottom + 30), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        status = f"Known: {len(set(db['names']))} people | Faces: {len(face_locations)} | L=learn TAB=select Q=quit"
        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == 9:  # TAB
            if face_locations:
                selected_face_idx = (selected_face_idx + 1) % len(face_locations)
        elif key == ord("l"):
            if face_locations and selected_face_idx < len(face_encodings):
                encoding = face_encodings[selected_face_idx]
                existing_match = None
                if db["encodings"]:
                    distances = face_recognition.face_distance(db["encodings"], encoding)
                    best_idx = np.argmin(distances)
                    if distances[best_idx] < TOLERANCE:
                        conf = max(0.0, 1.0 - distances[best_idx]) * 100
                        existing_match = (db["names"][best_idx], conf)
                name = get_name_from_gui(frame, existing_match)
                if name:
                    db["encodings"].append(encoding)
                    db["names"].append(name)
                    save_database(db)
                    save_face_image(name, frame, face_locations[selected_face_idx])
                    count = db["names"].count(name)
                    print(f"Learned face for '{name}' ({count} samples, {len(db['encodings'])} total encodings)")

    cap.release()
    cv2.destroyAllWindows()
    print("Goodbye!")


if __name__ == "__main__":
    main()
