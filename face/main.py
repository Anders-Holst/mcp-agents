import cv2
import face_recognition
import numpy as np
import os
import pickle
import threading
import sounddevice as sd
from faster_whisper import WhisperModel
from datetime import datetime
from piper import PiperVoice

KNOWN_FACES_DIR = "known_faces"
DB_FILE = os.path.join(KNOWN_FACES_DIR, "faces.pkl")
PIPER_MODEL_DIR = "piper_models"
TOLERANCE = 0.6
FRAME_SCALE = 0.5
UNKNOWN_ASK_COOLDOWN = 30  # seconds before asking the same unknown face again
SAMPLE_RATE = 16000
RECORD_SECONDS = 4


def load_database():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "names": []}


def save_database(db):
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)


def save_face_image(name, frame, face_location):
    top, right, bottom, left = face_location
    face_img = frame[top:bottom, left:right]
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(person_dir, f"{timestamp}.jpg")
    cv2.imwrite(filepath, face_img)


def show_overlay(frame, lines, window="Face Recognition"):
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
        if key == 27:
            return None
        elif key == 13 or key == 10:
            if name.strip():
                return name.strip()
            elif existing_match:
                return existing_match[0]
            return None
        elif key == 8 or key == 127:
            name = name[:-1]
        elif 32 <= key <= 126:
            name += chr(key)


class VoiceEngine:
    def __init__(self):
        self.whisper_model = None
        self.piper_voice = None
        self._loading = False
        self._ready = False

    def load(self):
        """Load models in background thread."""
        self._loading = True
        t = threading.Thread(target=self._load_models, daemon=True)
        t.start()

    def _load_models(self):
        print("Loading faster-whisper model (base)...")
        self.whisper_model = WhisperModel("base", compute_type="int8")
        print("Faster-whisper model loaded.")

        print("Loading piper voice model...")
        self._ensure_piper_model()
        model_path = os.path.join(PIPER_MODEL_DIR, "en_US-lessac-medium.onnx")
        self.piper_voice = PiperVoice.load(model_path)
        print("Piper voice loaded.")

        self._ready = True
        self._loading = False

    def _ensure_piper_model(self):
        os.makedirs(PIPER_MODEL_DIR, exist_ok=True)
        model_path = os.path.join(PIPER_MODEL_DIR, "en_US-lessac-medium.onnx")
        config_path = model_path + ".json"
        if not os.path.exists(model_path):
            print("Downloading piper voice model...")
            import urllib.request
            base = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/"
            urllib.request.urlretrieve(base + "en_US-lessac-medium.onnx", model_path)
            urllib.request.urlretrieve(base + "en_US-lessac-medium.onnx.json", config_path)
            print("Piper model downloaded.")

    @property
    def ready(self):
        return self._ready

    def speak(self, text):
        """Synthesize speech and play it."""
        if not self._ready:
            print(f"[TTS not ready] Would say: {text}")
            return
        print(f"[Speaking] {text}")
        sr = self.piper_voice.config.sample_rate
        chunks = []
        for audio_chunk in self.piper_voice.synthesize(text):
            chunks.append(audio_chunk.audio_float_array)
        audio = np.concatenate(chunks)
        sd.play(audio, sr)
        sd.wait()

    def listen(self, seconds=RECORD_SECONDS, on_segment=None):
        """Record from microphone and transcribe with streaming segments.
        on_segment(text_so_far) is called as each segment is decoded."""
        if not self._ready:
            print("[STT not ready]")
            return None
        print(f"[Listening for {seconds}s...]")
        audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=1, dtype="float32")
        sd.wait()
        audio = audio.flatten()
        # faster-whisper returns a generator of segments
        segments, info = self.whisper_model.transcribe(audio, beam_size=5)
        print(f"[Detected language: {info.language} ({info.language_probability:.0%})]")
        full_text = ""
        for segment in segments:
            full_text += segment.text
            print(f"  [Segment {segment.start:.1f}s-{segment.end:.1f}s] {segment.text.strip()}")
            if on_segment:
                on_segment(full_text.strip())
        text = full_text.strip()
        print(f"[Heard] {text}")
        return text


def extract_name(text):
    """Try to extract a name from a spoken response."""
    if not text:
        return None
    # Remove common filler phrases
    text = text.strip().strip(".")
    for prefix in ["my name is", "i'm", "i am", "they call me", "it's", "its",
                   "hi i'm", "hi i am", "hello i'm", "hello i am",
                   "hey i'm", "hey i am"]:
        lower = text.lower()
        if lower.startswith(prefix):
            text = text[len(prefix):].strip().strip(".,!")
            break
    # Take first word as name if multi-word
    name = text.split()[0] if text else None
    if name:
        name = name.strip(".,!?").capitalize()
    return name if name and len(name) > 1 else None


def main():
    db = load_database()
    voice = VoiceEngine()
    voice.load()

    print(f"Loaded {len(db['names'])} known face(s): {set(db['names'])}")
    print()
    print("Controls:")
    print("  L     - Learn/label face (keyboard)")
    print("  V     - Voice-ask the selected unknown face")
    print("  A     - Toggle auto-ask mode for unknown faces")
    print("  TAB   - Cycle selection between detected faces")
    print("  D     - Delete face database")
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
    face_confidences = []
    process_frame = True
    auto_ask = True
    # Track when we last asked about an unknown face encoding to avoid spam
    asked_recently = {}  # encoding_hash -> timestamp
    voice_status = ""
    voice_busy = False

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

        # Auto-ask unknown faces
        if auto_ask and voice.ready and not voice_busy:
            for i, (name, encoding) in enumerate(zip(face_names, face_encodings)):
                if name == "Unknown":
                    enc_hash = hash(encoding.tobytes())
                    now = datetime.now().timestamp()
                    if enc_hash not in asked_recently or (now - asked_recently[enc_hash]) > UNKNOWN_ASK_COOLDOWN:
                        asked_recently[enc_hash] = now
                        voice_busy = True
                        voice_status = "Asking..."
                        selected_face_idx = i

                        def _do_voice_ask(enc=encoding, loc=face_locations[i], frm=frame.copy()):
                            nonlocal voice_busy, voice_status, db
                            voice.speak("Hello! I don't think we've met. What is your name?")
                            voice_status = "Listening..."
                            def _on_seg(text):
                                nonlocal voice_status
                                voice_status = f"Hearing: {text}"
                            response = voice.listen(seconds=4, on_segment=_on_seg)
                            extracted = extract_name(response)
                            if extracted:
                                voice.speak(f"Nice to meet you, {extracted}!")
                                db["encodings"].append(enc)
                                db["names"].append(extracted)
                                save_database(db)
                                save_face_image(extracted, frm, loc)
                                voice_status = f"Learned: {extracted}"
                                print(f"Voice-learned '{extracted}'")
                            else:
                                voice.speak("Sorry, I didn't catch that.")
                                voice_status = "Didn't catch name"
                            voice_busy = False

                        threading.Thread(target=_do_voice_ask, daemon=True).start()
                        break

        # Draw results
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

        # Status bar
        voice_tag = " | Voice: loading..." if not voice.ready else ""
        auto_tag = " | AUTO-ASK ON" if auto_ask else ""
        if voice_status and voice.ready:
            voice_tag = f" | {voice_status}"
        status = f"Known: {len(set(db['names']))} people | Faces: {len(face_locations)}{auto_tag}{voice_tag}"
        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        keys_hint = "L=learn V=voice-ask A=auto-ask TAB=select D=delete Q=quit"
        h = frame.shape[0]
        cv2.putText(frame, keys_hint, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == 9:  # TAB
            if face_locations:
                selected_face_idx = (selected_face_idx + 1) % len(face_locations)
        elif key == ord("a"):
            auto_ask = not auto_ask
            voice_status = "Auto-ask " + ("enabled" if auto_ask else "disabled")
            print(f"Auto-ask mode: {'ON' if auto_ask else 'OFF'}")
        elif key == ord("v"):
            # Manual voice ask for selected face
            if face_locations and selected_face_idx < len(face_encodings) and voice.ready and not voice_busy:
                voice_busy = True
                voice_status = "Asking..."
                enc = face_encodings[selected_face_idx]
                loc = face_locations[selected_face_idx]
                frm = frame.copy()

                def _do_manual_voice(enc=enc, loc=loc, frm=frm):
                    nonlocal voice_busy, voice_status, db
                    voice.speak("Hello! What is your name?")
                    voice_status = "Listening..."
                    def _on_seg(text):
                        nonlocal voice_status
                        voice_status = f"Hearing: {text}"
                    response = voice.listen(seconds=4, on_segment=_on_seg)
                    extracted = extract_name(response)
                    if extracted:
                        voice.speak(f"Nice to meet you, {extracted}!")
                        db["encodings"].append(enc)
                        db["names"].append(extracted)
                        save_database(db)
                        save_face_image(extracted, frm, loc)
                        voice_status = f"Learned: {extracted}"
                        print(f"Voice-learned '{extracted}'")
                    else:
                        voice.speak("Sorry, I didn't catch that.")
                        voice_status = "Didn't catch name"
                    voice_busy = False

                threading.Thread(target=_do_manual_voice, daemon=True).start()
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
        elif key == ord("d"):
            # Show confirmation overlay
            show_overlay(frame, [
                ("Delete all known faces? Y/N", (0, 0, 255), 0.8),
            ])
            confirm = cv2.waitKey(0) & 0xFF
            if confirm == ord("y"):
                db = {"encodings": [], "names": []}
                save_database(db)
                print("Database cleared!")
                voice_status = "Database cleared"

    cap.release()
    cv2.destroyAllWindows()
    print("Goodbye!")


if __name__ == "__main__":
    main()
