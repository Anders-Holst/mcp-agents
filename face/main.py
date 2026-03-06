import cv2
import face_recognition
import numpy as np
import os
import pickle
import time
import threading
import sounddevice as sd
from faster_whisper import WhisperModel
from datetime import datetime
from piper import PiperVoice
import onnxruntime as ort

EMOTION_MODEL_DIR = "emotion_model"
EMOTION_MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
EMOTION_LABELS = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"]

KNOWN_FACES_DIR = "known_faces"
DB_FILE = os.path.join(KNOWN_FACES_DIR, "faces.pkl")
PIPER_MODEL_DIR = "piper_models"
TOLERANCE = 0.6
FRAME_SCALE = 0.5
UNKNOWN_ASK_COOLDOWN = 30  # seconds before asking the same unknown face again
SAMPLE_RATE = 16000
RECORD_SECONDS = 4
VAD_THRESHOLD = 0.7       # Silero VAD speech probability threshold
VAD_SILENCE_MS = 600      # ms of silence after speech to trigger transcription
VAD_MAX_SPEECH_S = 10     # max seconds of speech before forced transcription
VAD_PRE_SPEECH_MS = 300   # ms of audio to keep before speech onset
CONVERSATION_DISPLAY_S = 6  # how long to show conversation text on screen
AUDIO_METER_DECAY = 0.92    # how fast the peak marker decays (0-1, higher = slower)
CONTINUOUS_LISTEN = True    # always-on listening (no key press needed)
NOISE_REDUCE = True         # apply noise reduction before transcription


class AudioMonitor:
    """Continuously monitors microphone input level using a callback stream."""

    def __init__(self):
        self.rms = 0.0        # current RMS level (0-1)
        self.peak = 0.0       # recent peak level (decaying)
        self.max_seen = 0.001 # max level ever seen (for auto-scaling)
        self._stream = None

    def start(self):
        blocksize = int(SAMPLE_RATE * 0.05)  # 50ms chunks

        def callback(indata, frames, time_info, status):
            rms = float(np.sqrt(np.mean(indata ** 2)))
            self.rms = rms
            if rms > self.max_seen:
                self.max_seen = rms
            if rms > self.peak:
                self.peak = rms
            else:
                self.peak = self.peak * AUDIO_METER_DECAY

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype='float32',
            blocksize=blocksize, callback=callback
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def draw_audio_meter(frame, audio_monitor, voice_engine=None):
    """Draw a vertical audio level meter and VAD probability bar on the left side."""
    h, w = frame.shape[:2]
    # Volume meter dimensions
    mx, my = 15, 70       # top-left corner
    mw, mh = 14, h - 140  # width, height

    max_val = max(audio_monitor.max_seen, 0.001)
    level = min(1.0, audio_monitor.rms / max_val)
    peak = min(1.0, audio_monitor.peak / max_val)

    # Background bar (dark)
    overlay = frame.copy()
    cv2.rectangle(overlay, (mx, my), (mx + mw, my + mh), (30, 30, 30), cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Level fill (green -> yellow -> red)
    bar_h = int(level * mh)
    if bar_h > 0:
        for y in range(bar_h):
            frac = y / mh
            if frac < 0.6:
                color = (0, 200, 0)       # green
            elif frac < 0.85:
                color = (0, 220, 220)     # yellow
            else:
                color = (0, 0, 255)       # red
            y_pos = my + mh - y
            cv2.line(frame, (mx + 1, y_pos), (mx + mw - 1, y_pos), color, 1)

    # Peak marker (white horizontal line)
    peak_y = my + mh - int(peak * mh)
    cv2.line(frame, (mx - 2, peak_y), (mx + mw + 2, peak_y), (255, 255, 255), 2)

    # Border
    cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (100, 100, 100), 1)

    # dB label
    db = 20 * np.log10(audio_monitor.rms + 1e-10)
    cv2.putText(frame, f"{db:.0f}dB", (mx - 2, my - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    cv2.putText(frame, "VOL", (mx - 2, my + mh + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

    # VAD probability meter (next to volume meter)
    if voice_engine:
        vx = mx + mw + 8  # position right of volume meter
        vw = 14

        # Background
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (vx, my), (vx + vw, my + mh), (30, 30, 30), cv2.FILLED)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

        # VAD probability fill
        vad_level = voice_engine.vad_prob
        vad_h = int(vad_level * mh)
        if vad_h > 0:
            # Color based on whether above threshold
            vad_color = (0, 200, 255) if vad_level >= VAD_THRESHOLD else (180, 100, 0)
            for y in range(vad_h):
                y_pos = my + mh - y
                cv2.line(frame, (vx + 1, y_pos), (vx + vw - 1, y_pos), vad_color, 1)

        # Threshold line (cyan dashed)
        thresh_y = my + mh - int(VAD_THRESHOLD * mh)
        cv2.line(frame, (vx - 3, thresh_y), (vx + vw + 3, thresh_y), (0, 255, 255), 2)
        cv2.putText(frame, f"{VAD_THRESHOLD:.1f}", (vx + vw + 5, thresh_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

        # Border
        cv2.rectangle(frame, (vx, my), (vx + vw, my + mh), (100, 100, 100), 1)

        # Label
        cv2.putText(frame, "VAD", (vx - 2, my + mh + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

        # Phase indicator
        phase = voice_engine.listen_phase
        if phase:
            phase_colors = {
                "waiting": (0, 200, 255),      # orange
                "recording": (0, 0, 255),       # red
                "transcribing": (255, 200, 0),  # blue-ish
            }
            phase_color = phase_colors.get(phase, (200, 200, 200))
            # Pulsing effect for recording
            if phase == "recording":
                pulse = abs(np.sin(time.time() * 5))
                phase_color = (0, 0, int(150 + 105 * pulse))
            px = vx + vw + 8
            cv2.putText(frame, phase.upper(), (px, my + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, phase_color, 2)

        # Show detected language
        lang = voice_engine.detected_language
        if lang:
            lang_prob = voice_engine.detected_language_prob
            px = vx + vw + 8
            lang_text = f"Lang: {lang} ({lang_prob:.0%})"
            cv2.putText(frame, lang_text, (px, my + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def load_database():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            data = pickle.load(f)
        # Migrate old databases without last_seen
        if "last_seen" not in data:
            data["last_seen"] = {}
        return data
    return {"encodings": [], "names": [], "last_seen": {}}


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


class EmotionDetector:
    def __init__(self):
        self.session = None
        self._ensure_model()

    def _ensure_model(self):
        os.makedirs(EMOTION_MODEL_DIR, exist_ok=True)
        model_path = os.path.join(EMOTION_MODEL_DIR, "emotion-ferplus-8.onnx")
        if not os.path.exists(model_path):
            print("Downloading emotion detection model...")
            import urllib.request
            urllib.request.urlretrieve(EMOTION_MODEL_URL, model_path)
            print("Emotion model downloaded.")
        self.session = ort.InferenceSession(model_path)

    def detect(self, face_bgr):
        """Detect emotion from a BGR face crop. Returns (label, confidence)."""
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        input_data = resized.astype(np.float32).reshape(1, 1, 64, 64)
        input_name = self.session.get_inputs()[0].name
        result = self.session.run(None, {input_name: input_data})
        scores = result[0][0]
        # softmax
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        idx = np.argmax(probs)
        return EMOTION_LABELS[idx], float(probs[idx])


def update_last_seen(db, name):
    """Update the last_seen timestamp for a person."""
    db["last_seen"][name] = datetime.now().timestamp()


def get_greeting(db, name, emotion=None):
    """Generate a personalized greeting based on history and emotion."""
    now = datetime.now()
    hour = now.hour
    time_greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"

    last_seen = db["last_seen"].get(name)
    if last_seen is None:
        # First time ever
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


GREETING_COOLDOWN = 60  # seconds before greeting the same person again


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
        self.vad_prob = 0.0       # current VAD speech probability (0-1)
        self.listen_phase = ""    # "", "waiting", "recording", "transcribing"
        self.detected_language = ""      # last detected language code
        self.detected_language_prob = 0  # last detected language probability
        self.audio_monitor = None # set externally to pause/resume during listening
        self._voice_lock = threading.Lock()  # prevent concurrent listen/speak

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
        """Synthesize speech and play it, streaming audio as it's generated."""
        if not self._ready:
            print(f"[TTS not ready] Would say: {text}")
            return
        with self._voice_lock:
            self._speak_impl(text)

    def _speak_impl(self, text):
        print(f"[Speaking] {text}")
        sr = self.piper_voice.config.sample_rate
        buffer = []
        buffer_event = threading.Event()
        finished = threading.Event()

        def callback(outdata, frames, time_info, status):
            if buffer:
                chunk = buffer.pop(0)
                if len(chunk) < frames:
                    outdata[:len(chunk), 0] = chunk
                    outdata[len(chunk):, 0] = 0
                else:
                    outdata[:, 0] = chunk[:frames]
                    if len(chunk) > frames:
                        buffer.insert(0, chunk[frames:])
            else:
                outdata[:, 0] = 0
                if finished.is_set():
                    raise sd.CallbackStop

        stream = sd.OutputStream(
            samplerate=sr, channels=1, dtype="float32",
            blocksize=4096, callback=callback,
        )
        stream.start()
        for audio_chunk in self.piper_voice.synthesize(text):
            buffer.append(audio_chunk.audio_float_array)
        finished.set()
        while stream.active:
            sd.sleep(50)
        stream.close()

    def listen(self, seconds=RECORD_SECONDS, on_segment=None):
        """VAD-based listening: waits for speech, records until silence, then transcribes.
        Falls back to fixed-length recording if VAD is unavailable.
        on_segment(text_so_far) is called as each segment is decoded."""
        if not self._ready:
            print("[STT not ready]")
            return None

        if not self._voice_lock.acquire(timeout=1):
            print("[Voice engine busy, skipping listen]")
            return None
        try:
            try:
                return self._listen_vad(on_segment=on_segment)
            except Exception as e:
                print(f"[VAD listen failed: {e}, falling back to fixed recording]")
                return self._listen_fixed(seconds=seconds, on_segment=on_segment)
        finally:
            self._voice_lock.release()

    def _listen_vad(self, on_segment=None):
        """Listen using Voice Activity Detection for low-latency speech capture."""
        import torch
        import collections

        print("[Listening... (speak now)]")
        self.listen_phase = "waiting"
        self.vad_prob = 0.0

        # Load Silero VAD model (cached after first load)
        if not hasattr(self, '_vad_model'):
            print("[Loading Silero VAD model...]")
            self._vad_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad', model='silero_vad',
                trust_repo=True
            )
            print("[Silero VAD loaded.]")

        vad_model = self._vad_model
        vad_model.reset_states()

        chunk_ms = 32  # VAD needs at least 512 samples at 16kHz
        chunk_samples = 512  # exactly 512 samples = 32ms
        silence_chunks_needed = int(VAD_SILENCE_MS / chunk_ms)
        max_chunks = int(VAD_MAX_SPEECH_S * 1000 / chunk_ms)
        pre_speech_chunks = int(VAD_PRE_SPEECH_MS / chunk_ms)

        # Ring buffer for pre-speech audio
        pre_buffer = collections.deque(maxlen=pre_speech_chunks)
        speech_chunks = []
        silence_count = 0
        speech_started = False
        total_chunks = 0

        # Use a blocking stream for chunk-by-chunk reading
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype='float32',
            blocksize=chunk_samples
        )
        stream.start()

        try:
            while True:
                data, _ = stream.read(chunk_samples)
                chunk = data[:, 0]  # mono
                total_chunks += 1

                # Run VAD on this chunk
                tensor = torch.from_numpy(chunk)
                speech_prob = vad_model(tensor, SAMPLE_RATE).item()
                self.vad_prob = speech_prob

                if not speech_started:
                    pre_buffer.append(chunk.copy())
                    if speech_prob >= VAD_THRESHOLD:
                        speech_started = True
                        self.listen_phase = "recording"
                        silence_count = 0
                        # Include pre-speech buffer
                        speech_chunks.extend(pre_buffer)
                        speech_chunks.append(chunk.copy())
                        print("[Speech detected]")
                        if on_segment:
                            on_segment("(listening...)")
                    # Timeout: if no speech after 8 seconds, give up
                    elif total_chunks * chunk_ms > 8000:
                        print("[No speech detected, giving up]")
                        return None
                else:
                    speech_chunks.append(chunk.copy())
                    if speech_prob < VAD_THRESHOLD:
                        silence_count += 1
                        if silence_count >= silence_chunks_needed:
                            print(f"[End of speech detected ({len(speech_chunks) * chunk_ms}ms)]")
                            break
                    else:
                        silence_count = 0

                    if len(speech_chunks) >= max_chunks:
                        print("[Max speech length reached]")
                        break
        finally:
            stream.stop()
            stream.close()

        if not speech_chunks:
            self.listen_phase = ""
            self.vad_prob = 0.0
            return None

        self.listen_phase = "transcribing"
        self.vad_prob = 0.0
        audio = np.concatenate(speech_chunks)

        # Noise reduction
        if NOISE_REDUCE:
            try:
                import noisereduce as nr
                audio = nr.reduce_noise(y=audio, sr=SAMPLE_RATE, stationary=True,
                                        prop_decrease=0.75)
            except Exception as e:
                print(f"[Noise reduction failed: {e}]")

        # Transcribe
        segments, info = self.whisper_model.transcribe(audio, beam_size=5)
        self.detected_language = info.language
        self.detected_language_prob = info.language_probability
        print(f"[Detected language: {info.language} ({info.language_probability:.0%})]")

        full_text = ""
        for segment in segments:
            full_text += segment.text
            print(f"  [Segment {segment.start:.1f}s-{segment.end:.1f}s] {segment.text.strip()}")
            if on_segment:
                on_segment(full_text.strip())
        text = full_text.strip()
        print(f"[Heard] {text}")
        self.listen_phase = ""
        return text

    def _listen_fixed(self, seconds=RECORD_SECONDS, on_segment=None):
        """Fallback: fixed-length recording and transcription."""
        self.listen_phase = "recording"
        print(f"[Listening for {seconds}s...]")
        audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=1, dtype="float32")
        sd.wait()
        audio = audio.flatten()
        self.listen_phase = "transcribing"

        if NOISE_REDUCE:
            try:
                import noisereduce as nr
                audio = nr.reduce_noise(y=audio, sr=SAMPLE_RATE, stationary=True,
                                        prop_decrease=0.75)
            except Exception as e:
                print(f"[Noise reduction failed: {e}]")

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
        self.listen_phase = ""
        return text


class ContinuousListener:
    """Always-on background listener that detects speech via VAD and transcribes."""

    def __init__(self, voice_engine, on_heard=None):
        self.voice = voice_engine
        self.on_heard = on_heard  # callback(text) when speech is transcribed
        self._running = False
        self._paused = False  # pause while speaking or when voice_busy
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    @property
    def paused(self):
        return self._paused

    @paused.setter
    def paused(self, val):
        self._paused = val

    def _run(self):
        # Wait for voice engine to be ready
        while self._running and not self.voice.ready:
            time.sleep(0.5)

        print("[Continuous listener started]")
        while self._running:
            if self._paused:
                time.sleep(0.1)
                continue

            try:
                text = self.voice.listen()
                # Re-check paused after listen returns (may have been paused during listen)
                if text and self.on_heard and not self._paused:
                    self.on_heard(text)
                # Small delay between listen cycles to avoid tight loop
                time.sleep(0.3)
            except Exception as e:
                print(f"[Continuous listener error: {e}]")
                time.sleep(1)


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


def draw_listening_indicator(frame, is_listening, voice_status):
    """Draw a pulsing listening indicator and status on the frame."""
    if not is_listening:
        return
    h, w = frame.shape[:2]
    # Pulsing red circle (pulses using time)
    pulse = int(abs(np.sin(time.time() * 4)) * 8) + 12
    cx, cy = w - 40, 50
    cv2.circle(frame, (cx, cy), pulse, (0, 0, 255), -1)
    cv2.circle(frame, (cx, cy), pulse + 2, (0, 0, 255), 2)
    cv2.putText(frame, "LISTENING", (cx - 75, cy + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def draw_conversation(frame, conversation_lines):
    """Draw conversation history (what was said/heard) at the bottom of the frame."""
    if not conversation_lines:
        return
    h, w = frame.shape[:2]
    now = time.time()
    # Filter to recent lines
    active = [(role, text, ts) for role, text, ts in conversation_lines
              if now - ts < CONVERSATION_DISPLAY_S]
    if not active:
        return

    # Draw semi-transparent background
    line_h = 30
    box_h = len(active) * line_h + 20
    y_start = h - 50 - box_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, y_start), (w - 10, y_start + box_h), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, (role, text, ts) in enumerate(active):
        y = y_start + 22 + i * line_h
        age = now - ts
        # Fade out in last second
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


def main():
    db = load_database()
    voice = VoiceEngine()
    voice.load()
    emotion_detector = EmotionDetector()
    audio_monitor = AudioMonitor()
    audio_monitor.start()
    voice.audio_monitor = audio_monitor

    print(f"Loaded {len(db['names'])} known face(s): {set(db['names'])}")
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

    selected_face_idx = 0
    face_locations = []
    face_encodings = []
    face_names = []
    face_confidences = []
    face_emotions = []
    process_frame = True
    auto_ask = True
    # Track when we last asked about an unknown face encoding to avoid spam
    asked_recently = {}  # encoding_hash -> timestamp
    # Track when we last greeted a known person
    greeted_recently = {}  # name -> timestamp
    voice_status = ""
    voice_busy = False
    is_listening = False
    continuous_listen_enabled = CONTINUOUS_LISTEN
    # Conversation display: list of (role, text, timestamp)
    # role: "system" (what we said), "heard" (what user said)
    conversation_lines = []

    def add_conversation(role, text):
        conversation_lines.append((role, text, time.time()))
        # Keep only recent entries
        cutoff = time.time() - CONVERSATION_DISPLAY_S * 2
        while conversation_lines and conversation_lines[0][2] < cutoff:
            conversation_lines.pop(0)

    def do_speak_and_show(text):
        add_conversation("system", text)
        if continuous_listener:
            continuous_listener.paused = True
        voice.speak(text)
        if continuous_listener and continuous_listen_enabled:
            continuous_listener.paused = False

    def make_on_segment():
        """Create an on_segment callback that updates voice_status and shows live transcription."""
        def _on_seg(text):
            nonlocal voice_status
            voice_status = f"Hearing: {text}"
        return _on_seg

    # Language-aware response templates
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
        """Generate a language-aware response to heard speech."""
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
        """Handle speech detected by continuous listener."""
        nonlocal voice_busy, voice_status
        if voice_busy or not text:
            return
        voice_busy = True
        if continuous_listener:
            continuous_listener.paused = True
        # Find who's talking
        person = None
        for name in face_names:
            if name != "Unknown":
                person = name
                break
        lang = voice.detected_language or "?"
        lang_prob = voice.detected_language_prob or 0
        add_conversation("heard", f"[{lang} {lang_prob:.0%}] {text}")
        voice_status = f'Heard [{lang}]: "{text}"'
        response = get_response(text, person)
        if response:
            do_speak_and_show(response)
        voice_status = ""
        voice_busy = False
        if continuous_listener and continuous_listen_enabled:
            continuous_listener.paused = False

    # Set up continuous listener
    continuous_listener = None
    if CONTINUOUS_LISTEN:
        continuous_listener = ContinuousListener(voice, on_heard=handle_heard_speech)
        continuous_listener.start()
        print("[Continuous listening enabled - speak anytime]")

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

            # Detect emotions for each face
            face_emotions = []
            for top, right, bottom, left in face_locations:
                face_roi = frame[top:bottom, left:right]
                if face_roi.size > 0:
                    try:
                        label, conf = emotion_detector.detect(face_roi)
                        face_emotions.append(label)
                    except Exception:
                        face_emotions.append("neutral")
                else:
                    face_emotions.append("neutral")

            if selected_face_idx >= len(face_locations):
                selected_face_idx = 0

        process_frame = not process_frame

        # Greet known faces (then listen for a response)
        if voice.ready and not voice_busy:
            for i, name in enumerate(face_names):
                if name != "Unknown":
                    now = datetime.now().timestamp()
                    if name not in greeted_recently or (now - greeted_recently[name]) > GREETING_COOLDOWN:
                        greeted_recently[name] = now
                        emotion = face_emotions[i] if i < len(face_emotions) else None
                        greeting = get_greeting(db, name, emotion)
                        update_last_seen(db, name)
                        save_database(db)
                        voice_busy = True
                        voice_status = f"Greeting {name}..."

                        def _do_greet(text=greeting, person=name):
                            nonlocal voice_busy, voice_status, is_listening
                            if continuous_listener:
                                continuous_listener.paused = True
                            do_speak_and_show(text)
                            if not continuous_listen_enabled:
                                # Only manually listen after greeting if continuous is off
                                is_listening = True
                                voice_status = "Listening..."
                                response = voice.listen(on_segment=make_on_segment())
                                is_listening = False
                                if response:
                                    add_conversation("heard", response)
                                    resp = get_response(response, person)
                                    if resp:
                                        do_speak_and_show(resp)
                            voice_status = ""
                            voice_busy = False
                            if continuous_listener and continuous_listen_enabled:
                                continuous_listener.paused = False

                        threading.Thread(target=_do_greet, daemon=True).start()
                        break

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
                            nonlocal voice_busy, voice_status, db, is_listening
                            if continuous_listener:
                                continuous_listener.paused = True
                            do_speak_and_show("Hello! I don't think we've met. What is your name?")
                            is_listening = True
                            voice_status = "Listening..."
                            response = voice.listen(on_segment=make_on_segment())
                            is_listening = False
                            if response:
                                add_conversation("heard", response)
                            extracted = extract_name(response)
                            if extracted:
                                do_speak_and_show(f"Nice to meet you, {extracted}!")
                                db["encodings"].append(enc)
                                db["names"].append(extracted)
                                save_database(db)
                                save_face_image(extracted, frm, loc)
                                voice_status = f"Learned: {extracted}"
                                print(f"Voice-learned '{extracted}'")
                            else:
                                do_speak_and_show("Sorry, I didn't catch that.")
                                voice_status = "Didn't catch name"
                            voice_busy = False
                            if continuous_listener and continuous_listen_enabled:
                                continuous_listener.paused = False

                        threading.Thread(target=_do_voice_ask, daemon=True).start()
                        break

        # Draw results
        for i, ((top, right, bottom, left), name, conf) in enumerate(zip(face_locations, face_names, face_confidences)):
            is_selected = (i == selected_face_idx)
            emotion = face_emotions[i] if i < len(face_emotions) else ""
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
            # Show emotion above the face box
            if emotion:
                cv2.putText(frame, emotion, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw audio level meter + VAD bar
        draw_audio_meter(frame, audio_monitor, voice)

        # Draw listening indicator
        draw_listening_indicator(frame, is_listening, voice_status)

        # Draw conversation display
        draw_conversation(frame, conversation_lines)

        # Status bar
        voice_tag = " | Voice: loading..." if not voice.ready else ""
        auto_tag = " | AUTO-ASK" if auto_ask else ""
        if continuous_listen_enabled:
            auto_tag += " | ALWAYS-ON"
        if voice_status and voice.ready:
            voice_tag = f" | {voice_status}"
        status = f"Known: {len(set(db['names']))} people | Faces: {len(face_locations)}{auto_tag}{voice_tag}"
        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        keys_hint = "L=learn V=voice-ask T=talk C=continuous A=auto-ask TAB=select D=delete Q=quit"
        h = frame.shape[0]
        cv2.putText(frame, keys_hint, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == 9:  # TAB
            if face_locations:
                selected_face_idx = (selected_face_idx + 1) % len(face_locations)
        elif key == ord("c"):
            continuous_listen_enabled = not continuous_listen_enabled
            if continuous_listener:
                continuous_listener.paused = not continuous_listen_enabled
            elif continuous_listen_enabled:
                continuous_listener = ContinuousListener(voice, on_heard=handle_heard_speech)
                continuous_listener.start()
            voice_status = "Continuous listen " + ("enabled" if continuous_listen_enabled else "disabled")
            print(f"Continuous listening: {'ON' if continuous_listen_enabled else 'OFF'}")
        elif key == ord("a"):
            auto_ask = not auto_ask
            voice_status = "Auto-ask " + ("enabled" if auto_ask else "disabled")
            print(f"Auto-ask mode: {'ON' if auto_ask else 'OFF'}")
        elif key == ord("t"):
            # Talk mode: listen and respond
            if voice.ready and not voice_busy:
                voice_busy = True
                # Find who we're talking to
                talk_name = None
                for name in face_names:
                    if name != "Unknown":
                        talk_name = name
                        break

                def _do_talk(person=talk_name):
                    nonlocal voice_busy, voice_status, is_listening
                    if continuous_listener:
                        continuous_listener.paused = True
                    is_listening = True
                    voice_status = "Listening..."
                    response = voice.listen(on_segment=make_on_segment())
                    is_listening = False
                    if response:
                        add_conversation("heard", response)
                        resp = get_response(response, person)
                        if resp:
                            do_speak_and_show(resp)
                        voice_status = ""
                    else:
                        voice_status = "Didn't hear anything"
                    voice_busy = False
                    if continuous_listener and continuous_listen_enabled:
                        continuous_listener.paused = False

                threading.Thread(target=_do_talk, daemon=True).start()
        elif key == ord("v"):
            # Manual voice ask for selected face
            if face_locations and selected_face_idx < len(face_encodings) and voice.ready and not voice_busy:
                voice_busy = True
                voice_status = "Asking..."
                enc = face_encodings[selected_face_idx]
                loc = face_locations[selected_face_idx]
                frm = frame.copy()

                def _do_manual_voice(enc=enc, loc=loc, frm=frm):
                    nonlocal voice_busy, voice_status, db, is_listening
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
                        do_speak_and_show(f"Nice to meet you, {extracted}!")
                        db["encodings"].append(enc)
                        db["names"].append(extracted)
                        save_database(db)
                        save_face_image(extracted, frm, loc)
                        voice_status = f"Learned: {extracted}"
                        print(f"Voice-learned '{extracted}'")
                    else:
                        do_speak_and_show("Sorry, I didn't catch that.")
                        voice_status = "Didn't catch name"
                    voice_busy = False
                    if continuous_listener and continuous_listen_enabled:
                        continuous_listener.paused = False

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
                db = {"encodings": [], "names": [], "last_seen": {}}
                save_database(db)
                print("Database cleared!")
                voice_status = "Database cleared"

    if continuous_listener:
        continuous_listener.stop()
    audio_monitor.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Goodbye!")


if __name__ == "__main__":
    main()
