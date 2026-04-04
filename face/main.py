import cv2
import numpy as np
import os
import time
import threading
import logging
from datetime import datetime

from face_tracker import FaceTracker, FaceDatabase, EmotionDetector, TrackedFace
from voice_input import VoiceInput, AudioMonitor, ContinuousListener
from voice_output import VoiceOutput

# --- Logging setup ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logger = logging.getLogger("face_app")
logger.setLevel(logging.DEBUG)
_fh = logging.FileHandler(LOG_FILE)
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(_fh)
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(_ch)


class EventLog:
    """In-memory ring buffer of recent log events for on-screen display."""

    def __init__(self, max_entries=200):
        self._entries = []
        self._max = max_entries
        self._lock = threading.Lock()

    def add(self, category, message, detail=None):
        ts = datetime.now().strftime("%H:%M:%S")
        entry = {"ts": ts, "cat": category, "msg": message}
        if detail:
            entry["detail"] = detail
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max:
                self._entries.pop(0)
        detail_str = f" | {detail}" if detail else ""
        logger.info(f"[{category.upper()}] {message}{detail_str}")

    def recent(self, n=12):
        with self._lock:
            return list(self._entries[-n:])


event_log = EventLog()

PIPER_MODEL_DIR = "piper_models"
UNKNOWN_ASK_COOLDOWN = 30
CONVERSATION_DISPLAY_S = 6
CONTINUOUS_LISTEN = True
GREETING_COOLDOWN = 60


def draw_audio_meter(frame, audio_monitor, voice_engine=None):
    h, w = frame.shape[:2]
    mx, my = 15, 70
    mw, mh = 14, h - 140

    max_val = max(audio_monitor.max_seen, 0.001)
    level = min(1.0, audio_monitor.rms / max_val)
    peak = min(1.0, audio_monitor.peak / max_val)

    overlay = frame.copy()
    cv2.rectangle(overlay, (mx, my), (mx + mw, my + mh), (30, 30, 30), cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    bar_h = int(level * mh)
    if bar_h > 0:
        for y in range(bar_h):
            frac = y / mh
            if frac < 0.6:
                color = (0, 200, 0)
            elif frac < 0.85:
                color = (0, 220, 220)
            else:
                color = (0, 0, 255)
            y_pos = my + mh - y
            cv2.line(frame, (mx + 1, y_pos), (mx + mw - 1, y_pos), color, 1)

    peak_y = my + mh - int(peak * mh)
    cv2.line(frame, (mx - 2, peak_y), (mx + mw + 2, peak_y), (255, 255, 255), 2)
    cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (100, 100, 100), 1)

    db = 20 * np.log10(audio_monitor.rms + 1e-10)
    cv2.putText(frame, f"{db:.0f}dB", (mx - 2, my - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    cv2.putText(frame, "VOL", (mx - 2, my + mh + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

    if voice_engine:
        vx = mx + mw + 8
        vw = 14

        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (vx, my), (vx + vw, my + mh), (30, 30, 30), cv2.FILLED)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

        vad_level = voice_engine.vad_prob
        vad_h = int(vad_level * mh)
        if vad_h > 0:
            vad_color = (0, 200, 255) if vad_level >= voice_engine.vad_threshold else (180, 100, 0)
            for y in range(vad_h):
                y_pos = my + mh - y
                cv2.line(frame, (vx + 1, y_pos), (vx + vw - 1, y_pos), vad_color, 1)

        thresh_y = my + mh - int(voice_engine.vad_threshold * mh)
        cv2.line(frame, (vx - 3, thresh_y), (vx + vw + 3, thresh_y), (0, 255, 255), 2)
        cv2.putText(frame, f"{voice_engine.vad_threshold:.1f}", (vx + vw + 5, thresh_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        cv2.rectangle(frame, (vx, my), (vx + vw, my + mh), (100, 100, 100), 1)
        cv2.putText(frame, "VAD", (vx - 2, my + mh + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

        phase = voice_engine.listen_phase
        if phase:
            phase_colors = {
                "waiting": (0, 200, 255),
                "recording": (0, 0, 255),
                "transcribing": (255, 200, 0),
            }
            phase_color = phase_colors.get(phase, (200, 200, 200))
            if phase == "recording":
                pulse = abs(np.sin(time.time() * 5))
                phase_color = (0, 0, int(150 + 105 * pulse))
            px = vx + vw + 8
            cv2.putText(frame, phase.upper(), (px, my + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, phase_color, 2)

        lang = voice_engine.detected_language
        if lang:
            lang_prob = voice_engine.detected_language_prob
            px = vx + vw + 8
            lang_text = f"Lang: {lang} ({lang_prob:.0%})"
            cv2.putText(frame, lang_text, (px, my + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def get_greeting(face_db, name, emotion=None):
    """Generate a personalized greeting based on history and emotion."""
    now = datetime.now()
    hour = now.hour
    time_greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"

    last_seen = face_db.get_last_seen(name)
    if last_seen is None:
        greeting = f"Nice to meet you, {name}!"
    else:
        elapsed = now.timestamp() - last_seen
        if elapsed < 60:
            greeting = f"Hey again, {name}!"
        elif elapsed < 3600:
            mins = int(elapsed / 60)
            greeting = f"Welcome back, {name}! It's been {mins} minutes."
        elif elapsed < 86400:
            hours = int(elapsed / 3600)
            greeting = f"{time_greeting}, {name}! Haven't seen you in {hours} hour{'s' if hours > 1 else ''}."
        else:
            days = int(elapsed / 86400)
            greeting = f"{time_greeting}, {name}! It's been {days} day{'s' if days > 1 else ''} since I last saw you."

    emotion_comments = {
        "happy": "You look happy today!",
        "sad": "You seem a bit down. Hope your day gets better!",
        "angry": "You look a bit upset. Everything okay?",
        "surprise": "You look surprised!",
        "fear": "You look a bit worried.",
        "disgust": "Rough day?",
        "neutral": "",
    }
    if emotion and emotion in emotion_comments and emotion_comments[emotion]:
        greeting += " " + emotion_comments[emotion]

    return greeting


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


PIPER_MODEL_DIR = "piper_models"  # kept for reference, VoiceOutput has its own default


def extract_name(text):
    if not text:
        return None
    text = text.strip().strip(".")
    for prefix in ["my name is", "i'm", "i am", "they call me", "it's", "its",
                   "hi i'm", "hi i am", "hello i'm", "hello i am",
                   "hey i'm", "hey i am"]:
        lower = text.lower()
        if lower.startswith(prefix):
            text = text[len(prefix):].strip().strip(".,!")
            break
    name = text.split()[0] if text else None
    if name:
        name = name.strip(".,!?").capitalize()
    return name if name and len(name) > 1 else None


def draw_listening_indicator(frame, is_listening, voice_status):
    if not is_listening:
        return
    h, w = frame.shape[:2]
    pulse = int(abs(np.sin(time.time() * 4)) * 8) + 12
    cx, cy = w - 40, 50
    cv2.circle(frame, (cx, cy), pulse, (0, 0, 255), -1)
    cv2.circle(frame, (cx, cy), pulse + 2, (0, 0, 255), 2)
    cv2.putText(frame, "LISTENING", (cx - 75, cy + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def draw_conversation(frame, conversation_lines):
    if not conversation_lines:
        return
    h, w = frame.shape[:2]
    now = time.time()
    active = [(role, text, ts) for role, text, ts in conversation_lines
              if now - ts < CONVERSATION_DISPLAY_S]
    if not active:
        return

    line_h = 30
    box_h = len(active) * line_h + 20
    y_start = h - 50 - box_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, y_start), (w - 10, y_start + box_h), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, (role, text, ts) in enumerate(active):
        y = y_start + 22 + i * line_h
        age = now - ts
        alpha = min(1.0, (CONVERSATION_DISPLAY_S - age) / 1.0)
        if role == "system":
            color = (int(200 * alpha), int(200 * alpha), int(255 * alpha))
            prefix = ">> "
        elif role == "heard":
            color = (int(100 * alpha), int(255 * alpha), int(100 * alpha))
            prefix = "<< "
        else:
            color = (int(200 * alpha), int(200 * alpha), int(200 * alpha))
            prefix = ""
        cv2.putText(frame, prefix + text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


LOG_CATEGORY_COLORS = {
    "face":     (255, 200, 100),
    "voice":    (100, 255, 100),
    "action":   (100, 200, 255),
    "system":   (200, 200, 200),
    "emotion":  (200, 100, 255),
    "reasoning":(100, 255, 255),
}


def draw_event_log_window(event_log_inst, max_lines=30, width=700, height=600):
    """Render the event log into a separate OpenCV window."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (25, 25, 25)  # dark background

    entries = event_log_inst.recent(max_lines)

    # Title bar
    cv2.rectangle(canvas, (0, 0), (width, 30), (40, 40, 40), cv2.FILLED)
    cv2.putText(canvas, "EVENT LOG", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    entry_count = len(event_log_inst.recent(200))
    cv2.putText(canvas, f"({entry_count} events)", (width - 130, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    if not entries:
        cv2.putText(canvas, "No events yet...", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
        cv2.imshow("Event Log", canvas)
        return

    line_h = 19
    y_start = 40

    for i, entry in enumerate(entries):
        y = y_start + i * line_h
        if y + line_h > height:
            break
        cat = entry["cat"]
        color = LOG_CATEGORY_COLORS.get(cat, (180, 180, 180))
        ts = entry["ts"]
        msg = entry["msg"]
        detail = entry.get("detail", "")

        # Timestamp
        cv2.putText(canvas, ts, (8, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (100, 100, 100), 1)

        # Category tag
        tag = f"[{cat.upper()}]"
        cv2.putText(canvas, tag, (72, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1)

        # Message
        msg_x = 150
        max_msg_chars = 40
        display_msg = msg if len(msg) <= max_msg_chars else msg[:max_msg_chars - 2] + ".."
        cv2.putText(canvas, display_msg, (msg_x, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1)

        # Detail (dimmer, to the right)
        if detail:
            detail_x = msg_x + max_msg_chars * 8
            max_detail = 35
            display_detail = detail if len(detail) <= max_detail else detail[:max_detail - 2] + ".."
            cv2.putText(canvas, display_detail, (detail_x, y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (130, 130, 130), 1)

        # Subtle separator line
        cv2.line(canvas, (8, y + line_h - 1), (width - 8, y + line_h - 1), (40, 40, 40), 1)

    cv2.imshow("Event Log", canvas)


def main():
    # --- Face tracker setup ---
    face_db = FaceDatabase()
    face_db.load()
    emotion_detector = EmotionDetector()
    tracker = FaceTracker(face_db, emotion_detector,
                          on_event=lambda cat, msg, detail=None: event_log.add(cat, msg, detail))

    # --- Voice setup ---
    voice = VoiceInput()
    voice.load()
    tts = VoiceOutput()
    tts.load()
    audio_monitor = AudioMonitor()
    audio_monitor.start()

    event_log.add("system", f"App started, {len(face_db.known_names)} known people",
                  f"known: {', '.join(face_db.known_names) if face_db.known_names else 'none'}")
    event_log.add("system", f"Log file: {LOG_FILE}")
    print(f"Loaded {face_db.encoding_count} encodings for {len(face_db.known_names)} people: {face_db.known_names}")
    print()
    print("Controls:")
    print("  L     - Learn/label face (keyboard)")
    print("  V     - Voice-ask the selected unknown face")
    print("  T     - Talk (listen and respond)")
    print("  C     - Toggle continuous listening")
    print("  A     - Toggle auto-ask mode for unknown faces")
    print("  TAB   - Cycle selection between detected faces")
    print("  D     - Delete face database")
    print("  Q/ESC - Quit")
    print()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Recognition", 800, 600)
    cv2.namedWindow("Event Log", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Event Log", 700, 600)
    cv2.moveWindow("Event Log", 820, 0)

    selected_track_id = None  # Stable track ID instead of volatile index
    auto_ask = True
    asked_tracks = {}  # track_id -> timestamp (cooldown for auto-ask)
    greeted_recently = {}  # name -> timestamp
    voice_status = ""
    voice_busy = False
    is_listening = False
    continuous_listen_enabled = CONTINUOUS_LISTEN
    conversation_lines = []

    def add_conversation(role, text):
        conversation_lines.append((role, text, time.time()))
        cutoff = time.time() - CONVERSATION_DISPLAY_S * 2
        while conversation_lines and conversation_lines[0][2] < cutoff:
            conversation_lines.pop(0)

    def do_speak_and_show(text):
        add_conversation("system", text)
        if continuous_listener:
            continuous_listener.paused = True
        tts.speak(text)
        if continuous_listener and continuous_listen_enabled:
            continuous_listener.paused = False

    def make_on_segment():
        def _on_seg(text):
            nonlocal voice_status
            voice_status = f"Hearing: {text}"
        return _on_seg

    RESPONSES = {
        "en": {
            "greet": ["hello", "hi ", "hey"],
            "greet_r": "Hey {who}!",
            "how": ["how are you", "how're you", "how do you do"],
            "how_r": "I'm doing great, thanks for asking!",
            "thanks": ["thank", "thanks"],
            "thanks_r": "You're welcome!",
            "bye": ["bye", "goodbye", "see you"],
            "bye_r": "Goodbye {who}, see you later!",
            "time": ["what time", "what's the time"],
            "time_r": "It's {time}.",
            "who": ["who am i", "what's my name", "do you know me"],
            "who_known_r": "You're {who}, of course!",
            "who_unknown_r": "I don't think we've met yet.",
            "echo_r": "I heard you say: {text}",
        },
        "sv": {
            "greet": ["hej", "hallå", "tjena"],
            "greet_r": "Hej {who}!",
            "how": ["hur mår du", "hur är det", "läget"],
            "how_r": "Jag mår bra, tack för att du frågar!",
            "thanks": ["tack"],
            "thanks_r": "Varsågod!",
            "bye": ["hejdå", "vi ses", "adjö"],
            "bye_r": "Hejdå {who}, vi ses!",
            "time": ["vad är klockan", "vilken tid"],
            "time_r": "Klockan är {time}.",
            "who": ["vem är jag", "vet du vem jag är"],
            "who_known_r": "Du är {who}, såklart!",
            "who_unknown_r": "Jag tror inte vi har träffats.",
            "echo_r": "Jag hörde dig säga: {text}",
        },
        "de": {
            "greet": ["hallo", "hi ", "hey", "guten tag"],
            "greet_r": "Hallo {who}!",
            "how": ["wie geht", "wie bist du"],
            "how_r": "Mir geht es gut, danke der Nachfrage!",
            "thanks": ["danke"],
            "thanks_r": "Bitte schön!",
            "bye": ["tschüss", "auf wiedersehen", "bis bald"],
            "bye_r": "Tschüss {who}, bis später!",
            "time": ["wie spät", "wieviel uhr"],
            "time_r": "Es ist {time}.",
            "who": ["wer bin ich", "kennst du mich"],
            "who_known_r": "Du bist {who}, natürlich!",
            "who_unknown_r": "Ich glaube nicht, dass wir uns kennen.",
            "echo_r": "Ich habe gehört: {text}",
        },
    }

    def get_response(text, person=None):
        if not text:
            return None
        lang = voice.detected_language or "en"
        templates = RESPONSES.get(lang, RESPONSES["en"])
        lower = text.lower()
        who = person or ("there" if lang == "en" else "")
        time_str = datetime.now().strftime("%I:%M %p")

        if any(w in lower for w in templates["how"]):
            return templates["how_r"]
        elif any(w in lower for w in templates["thanks"]):
            return templates["thanks_r"]
        elif any(w in lower for w in templates["greet"]):
            return templates["greet_r"].format(who=who)
        elif any(w in lower for w in templates["bye"]):
            return templates["bye_r"].format(who=who)
        elif any(w in lower for w in templates["time"]):
            return templates["time_r"].format(time=time_str)
        elif any(w in lower for w in templates["who"]):
            if person:
                return templates["who_known_r"].format(who=who)
            else:
                return templates["who_unknown_r"]
        else:
            return templates["echo_r"].format(text=text)

    def handle_heard_speech(text):
        nonlocal voice_busy, voice_status
        if voice_busy or not text:
            return
        voice_busy = True
        if continuous_listener:
            continuous_listener.paused = True
        primary = tracker.get_primary_face()
        person = tracker.get_name(primary.track_id) if primary else None
        lang = voice.detected_language or "?"
        lang_prob = voice.detected_language_prob or 0
        event_log.add("voice", f"Speech detected ({lang} {lang_prob:.0%})",
                      f'speaker={person or "unknown"}, text="{text}"')
        add_conversation("heard", f"[{lang} {lang_prob:.0%}] {text}")
        voice_status = f'Heard [{lang}]: "{text}"'
        response = get_response(text, person)
        if response:
            event_log.add("action", f"Responding to {person or 'unknown'}",
                          f'"{response}"')
            do_speak_and_show(response)
        voice_status = ""
        voice_busy = False
        if continuous_listener and continuous_listen_enabled:
            continuous_listener.paused = False

    continuous_listener = None
    if CONTINUOUS_LISTEN:
        continuous_listener = ContinuousListener(voice, on_heard=handle_heard_speech)
        continuous_listener.start()
        print("[Continuous listening enabled - speak anytime]")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Face tracking (replaces all inline detection/recognition/emotion) ---
        faces = tracker.process_frame(frame)
        visible_faces = [f for f in faces if f.is_visible]

        # Ensure selected_track_id is valid
        if selected_track_id is not None:
            if not any(f.track_id == selected_track_id for f in visible_faces):
                selected_track_id = visible_faces[0].track_id if visible_faces else None
        elif visible_faces:
            selected_track_id = visible_faces[0].track_id

        # --- Greet known faces ---
        if (voice.ready and tts.ready) and not voice_busy:
            for face in visible_faces:
                name = tracker.get_name(face.track_id)
                if name:
                    now = time.time()
                    if name not in greeted_recently or (now - greeted_recently[name]) > GREETING_COOLDOWN:
                        last_seen_ts = face_db.get_last_seen(name)
                        if last_seen_ts:
                            event_log.add("reasoning", f"Greeting {name}: last seen {now - last_seen_ts:.0f}s ago, emotion={face.emotion}")
                        else:
                            event_log.add("reasoning", f"Greeting {name}: first encounter, emotion={face.emotion}")
                        greeted_recently[name] = now
                        greeting = get_greeting(face_db, name, face.emotion)
                        face_db.update_last_seen(name)
                        face_db.save()
                        voice_busy = True
                        voice_status = f"Greeting {name}..."
                        event_log.add("action", f"Speaking greeting to {name}", f'"{greeting}"')

                        def _do_greet(text=greeting, person=name):
                            nonlocal voice_busy, voice_status, is_listening
                            if continuous_listener:
                                continuous_listener.paused = True
                            do_speak_and_show(text)
                            if not continuous_listen_enabled:
                                is_listening = True
                                voice_status = "Listening..."
                                event_log.add("voice", f"Listening for response from {person}")
                                response = voice.listen(on_segment=make_on_segment())
                                is_listening = False
                                if response:
                                    add_conversation("heard", response)
                                    event_log.add("voice", f"Heard from {person}: {response}")
                                    resp = get_response(response, person)
                                    if resp:
                                        event_log.add("action", f"Responding", f'"{resp}"')
                                        do_speak_and_show(resp)
                            voice_status = ""
                            voice_busy = False
                            if continuous_listener and continuous_listen_enabled:
                                continuous_listener.paused = False

                        threading.Thread(target=_do_greet, daemon=True).start()
                        break

        # --- Auto-ask unknown faces (using stable track IDs) ---
        if auto_ask and (voice.ready and tts.ready) and not voice_busy:
            for face in visible_faces:
                if not tracker.is_recognized(face.track_id):
                    tid = face.track_id
                    if face.frames_visible < 3:
                        continue  # wait for 3 consecutive frames before asking
                    now = time.time()
                    if tid in asked_tracks and (now - asked_tracks[tid]) <= UNKNOWN_ASK_COOLDOWN:
                        continue
                    event_log.add("reasoning", f"Unknown face (track {tid}): auto-asking",
                                  f"visible for {face.frames_visible} frames")
                    asked_tracks[tid] = now
                    voice_busy = True
                    voice_status = "Asking..."
                    selected_track_id = tid

                    def _do_voice_ask(track_id=tid, frm=frame.copy()):
                        nonlocal voice_busy, voice_status, is_listening
                        if continuous_listener:
                            continuous_listener.paused = True
                        event_log.add("action", "Asking unknown face for name")
                        do_speak_and_show("Hello! I don't think we've met. What is your name?")
                        is_listening = True
                        voice_status = "Listening..."
                        response = voice.listen(on_segment=make_on_segment())
                        is_listening = False
                        if response:
                            add_conversation("heard", response)
                            event_log.add("voice", f"Heard: {response}")
                        extracted = extract_name(response)
                        if extracted:
                            event_log.add("face", f"Learned new face: {extracted}",
                                          f"from speech: '{response}'")
                            tracker.learn_face(track_id, extracted, frm)
                            do_speak_and_show(f"Nice to meet you, {extracted}!")
                            voice_status = f"Learned: {extracted}"
                        else:
                            event_log.add("voice", "Failed to extract name",
                                          f"raw speech: '{response}'")
                            do_speak_and_show("Sorry, I didn't catch that.")
                            voice_status = "Didn't catch name"
                        voice_busy = False
                        if continuous_listener and continuous_listen_enabled:
                            continuous_listener.paused = False

                    threading.Thread(target=_do_voice_ask, daemon=True).start()
                    break

        # --- Draw face boxes ---
        for face in visible_faces:
            is_selected = (face.track_id == selected_track_id)
            face_name = tracker.get_name(face.track_id)
            face_conf = tracker.get_confidence(face.track_id)
            color = (0, 255, 255) if is_selected else (0, 255, 0) if face_name else (0, 0, 255)
            thickness = 3 if is_selected else 2
            top, right, bottom, left = face.bbox

            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
            if face_name:
                label = f"{face_name} {face_conf:.0f}%"
            else:
                label = f"Unknown #{face.track_id}"
            if is_selected:
                label += " [sel]"
            cv2.rectangle(frame, (left, bottom), (right, bottom + 30), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            if face.emotion:
                cv2.putText(frame, face.emotion, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- Draw overlays ---
        draw_audio_meter(frame, audio_monitor, voice)
        draw_listening_indicator(frame, is_listening, voice_status)
        draw_conversation(frame, conversation_lines)
        draw_event_log_window(event_log)

        # --- Status bar ---
        voice_tag = " | Voice: loading..." if not (voice.ready and tts.ready) else ""
        auto_tag = " | AUTO-ASK" if auto_ask else ""
        if continuous_listen_enabled:
            auto_tag += " | ALWAYS-ON"
        if voice_status and (voice.ready and tts.ready):
            voice_tag = f" | {voice_status}"
        status = f"Known: {len(face_db.known_names)} people | Faces: {len(visible_faces)}{auto_tag}{voice_tag}"
        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        keys_hint = "L=learn V=voice-ask T=talk C=continuous A=auto-ask TAB=select D=delete Q=quit"
        h = frame.shape[0]
        cv2.putText(frame, keys_hint, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == 9:  # TAB - cycle through visible faces by stable ID
            if visible_faces:
                ids = [f.track_id for f in visible_faces]
                if selected_track_id in ids:
                    idx = (ids.index(selected_track_id) + 1) % len(ids)
                    selected_track_id = ids[idx]
                else:
                    selected_track_id = ids[0]
        elif key == ord("c"):
            continuous_listen_enabled = not continuous_listen_enabled
            if continuous_listener:
                continuous_listener.paused = not continuous_listen_enabled
            elif continuous_listen_enabled:
                continuous_listener = ContinuousListener(voice, on_heard=handle_heard_speech)
                continuous_listener.start()
            voice_status = "Continuous listen " + ("enabled" if continuous_listen_enabled else "disabled")
            event_log.add("action", f"Continuous listen: {'ON' if continuous_listen_enabled else 'OFF'}")
        elif key == ord("a"):
            auto_ask = not auto_ask
            voice_status = "Auto-ask " + ("enabled" if auto_ask else "disabled")
            event_log.add("action", f"Auto-ask: {'ON' if auto_ask else 'OFF'}")
        elif key == ord("t"):
            if (voice.ready and tts.ready) and not voice_busy:
                voice_busy = True
                primary = tracker.get_primary_face()
                talk_name = tracker.get_name(primary.track_id) if primary else None

                def _do_talk(person=talk_name):
                    nonlocal voice_busy, voice_status, is_listening
                    if continuous_listener:
                        continuous_listener.paused = True
                    event_log.add("voice", f"Talk mode: listening", f"talking_to={person or 'unknown'}")
                    is_listening = True
                    voice_status = "Listening..."
                    response = voice.listen(on_segment=make_on_segment())
                    is_listening = False
                    if response:
                        add_conversation("heard", response)
                        event_log.add("voice", f"Heard: {response}")
                        resp = get_response(response, person)
                        if resp:
                            event_log.add("action", f"Responding", f'"{resp}"')
                            do_speak_and_show(resp)
                        voice_status = ""
                    else:
                        event_log.add("voice", "No speech detected in talk mode")
                        voice_status = "Didn't hear anything"
                    voice_busy = False
                    if continuous_listener and continuous_listen_enabled:
                        continuous_listener.paused = False

                threading.Thread(target=_do_talk, daemon=True).start()
        elif key == ord("v"):
            if selected_track_id and (voice.ready and tts.ready) and not voice_busy:
                voice_busy = True
                voice_status = "Asking..."
                frm = frame.copy()
                ask_tid = selected_track_id

                def _do_manual_voice(track_id=ask_tid, frm=frm):
                    nonlocal voice_busy, voice_status, is_listening
                    if continuous_listener:
                        continuous_listener.paused = True
                    do_speak_and_show("Hello! What is your name?")
                    is_listening = True
                    voice_status = "Listening..."
                    response = voice.listen(on_segment=make_on_segment())
                    is_listening = False
                    if response:
                        add_conversation("heard", response)
                    extracted = extract_name(response)
                    if extracted:
                        tracker.learn_face(track_id, extracted, frm)
                        do_speak_and_show(f"Nice to meet you, {extracted}!")
                        voice_status = f"Learned: {extracted}"
                    else:
                        do_speak_and_show("Sorry, I didn't catch that.")
                        voice_status = "Didn't catch name"
                    voice_busy = False
                    if continuous_listener and continuous_listen_enabled:
                        continuous_listener.paused = False

                threading.Thread(target=_do_manual_voice, daemon=True).start()
        elif key == ord("l"):
            if selected_track_id:
                face = tracker.get_face_by_id(selected_track_id)
                if face:
                    existing_match = None
                    if tracker.is_recognized(face.track_id):
                        existing_match = (tracker.get_name(face.track_id), tracker.get_confidence(face.track_id))
                    name = get_name_from_gui(frame, existing_match)
                    if name:
                        tracker.learn_face(selected_track_id, name, frame)
                        event_log.add("face", f"Manually learned: {name}")
        elif key == ord("d"):
            show_overlay(frame, [
                ("Delete all known faces? Y/N", (0, 0, 255), 0.8),
            ])
            confirm = cv2.waitKey(0) & 0xFF
            if confirm == ord("y"):
                event_log.add("action", "Database cleared", f"was {len(face_db.known_names)} people")
                face_db.clear()
                voice_status = "Database cleared"

    event_log.add("system", "App shutting down")
    if continuous_listener:
        continuous_listener.stop()
    audio_monitor.stop()
    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"Session log saved to {LOG_FILE}")
    print(f"Log saved to {LOG_FILE}")
    print("Goodbye!")


if __name__ == "__main__":
    main()
