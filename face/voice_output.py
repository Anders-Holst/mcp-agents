"""
Voice output module: text-to-speech using piper.

Provides:
- VoiceOutput: synthesize text and play through speakers
- Typed event system with subscribe/unsubscribe
- Blocking and non-blocking speak modes

Can be run standalone:
    python voice_output.py "Hello, world!"
    python voice_output.py --interactive
"""

import collections
import os
import re
import time
import threading
import logging
import argparse
from enum import Enum, auto
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Callable, Union

import sounddevice as sd
from piper import PiperVoice

from events import EventDispatcher

logger = logging.getLogger("voice_output")

_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

PIPER_MODEL_DIR = os.path.join(_SOURCE_DIR, "piper_models")
PIPER_MODEL_NAME = "en_US-lessac-medium"

from languages_config import (
    get_language_models, get_language_pronunciations, get_default_language,
)

LANGUAGE_MODELS = get_language_models()
DEFAULT_LANGUAGE = get_default_language()
LANG_PRONUNCIATIONS = get_language_pronunciations()


# ---------------------------------------------------------------------------
# Event types and payloads
# ---------------------------------------------------------------------------

class TtsEventType(Enum):
    MODEL_LOADING = auto()
    MODEL_READY = auto()
    MODEL_LOAD_FAILED = auto()
    TTS_STARTED = auto()
    TTS_FINISHED = auto()
    TTS_BUSY = auto()       # speak() called while already speaking


@dataclass(frozen=True)
class ModelLoadingPayload:
    model_name: str


@dataclass(frozen=True)
class ModelReadyPayload:
    model_name: str


@dataclass(frozen=True)
class ModelLoadFailedPayload:
    model_name: str
    error: str


@dataclass(frozen=True)
class TtsStartedPayload:
    text: str
    tts_text: Optional[str] = None  # phonetic text if pronunciations were applied


@dataclass(frozen=True)
class TtsFinishedPayload:
    text: str
    duration_s: float


@dataclass(frozen=True)
class TtsBusyPayload:
    rejected_text: str


TtsEventPayload = Union[
    ModelLoadingPayload, ModelReadyPayload, ModelLoadFailedPayload,
    TtsStartedPayload, TtsFinishedPayload, TtsBusyPayload,
]


@dataclass(frozen=True)
class TtsEvent:
    type: TtsEventType
    timestamp: float
    payload: TtsEventPayload


TtsEventCallback = Callable[[TtsEvent], None]


# ---------------------------------------------------------------------------
# VoiceOutput
# ---------------------------------------------------------------------------

class VoiceOutput:
    """Text-to-speech engine using piper.

    Subscribe to TtsEvent callbacks via subscribe().
    """

    def __init__(self, *,
                 model_dir: str = PIPER_MODEL_DIR,
                 model_name: str = PIPER_MODEL_NAME,
                 language_models: Optional[dict] = None):
        self._model_dir = model_dir
        self._default_model = model_name
        self._language_models = dict(language_models or LANGUAGE_MODELS)
        self._lang_pron = LANG_PRONUNCIATIONS
        # Loaded PiperVoice instances keyed by model name
        self._voices: dict[str, PiperVoice] = {}
        self._model_lock = threading.Lock()  # protects _voices lazy loading
        self._ready = False
        self._load_error: Optional[str] = None
        self._speaking = False
        self._stop_event = threading.Event()
        self._interrupted = False
        self._lock = threading.Lock()
        self._dispatcher = EventDispatcher(owner="voice_output")

    # --- Event API ---

    def subscribe(self, callback: TtsEventCallback,
                  event_types: Optional[set] = None) -> Callable[[], None]:
        return self._dispatcher.subscribe(callback, event_types)

    def unsubscribe(self, callback: TtsEventCallback) -> bool:
        return self._dispatcher.unsubscribe(callback)

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def load_error(self) -> Optional[str]:
        """Error message from the last default-model load attempt, or None."""
        return self._load_error

    @property
    def speaking(self) -> bool:
        return self._speaking

    @speaking.setter
    def speaking(self, value: bool):
        self._speaking = value

    # --- Lifecycle ---

    def load(self):
        """Load TTS model in background thread."""
        threading.Thread(target=self._load, daemon=True).start()

    def load_sync(self):
        """Load TTS model synchronously (blocking)."""
        self._load()

    def _load(self):
        """Load the default model so the engine is ready quickly."""
        self._load_model(self._default_model)

    def _load_model(self, model_name: str):
        """Load a single piper model by name from disk.

        Never downloads — run `python download_models.py` beforehand.
        Thread-safe: uses _model_lock to prevent duplicate loads.
        """
        with self._model_lock:
            if model_name in self._voices:
                return
            self._emit(TtsEventType.MODEL_LOADING,
                       ModelLoadingPayload(model_name))
            model_path = os.path.join(self._model_dir, f"{model_name}.onnx")
            if not os.path.exists(model_path):
                msg = (f"Piper model {model_name!r} not found at "
                       f"{model_path} — run download_models.py")
                if model_name == self._default_model:
                    self._load_error = msg
                self._emit(TtsEventType.MODEL_LOAD_FAILED,
                           ModelLoadFailedPayload(model_name, msg))
                logger.error(msg)
                return
            try:
                logger.info(f"Loading piper model: {model_name}...")
                self._voices[model_name] = PiperVoice.load(model_path)
                self._ready = True
                self._emit(TtsEventType.MODEL_READY,
                           ModelReadyPayload(model_name))
                logger.info(f"Piper TTS loaded: {model_name}")
            except Exception as e:
                msg = f"Failed to load piper model {model_name}: {e}"
                if model_name == self._default_model:
                    self._load_error = msg
                self._emit(TtsEventType.MODEL_LOAD_FAILED,
                           ModelLoadFailedPayload(model_name, str(e)))
                logger.error(msg)

    def _get_voice(self, language: Optional[str] = None) -> Optional[PiperVoice]:
        """Return the PiperVoice for a language, or None if not loaded.

        Never downloads. Falls back to the default model if the requested
        language's model is not available on disk.
        """
        model_name = self._language_models.get(
            language, self._default_model) if language else self._default_model
        if model_name not in self._voices:
            self._load_model(model_name)
        if model_name not in self._voices:
            logger.warning(
                f"no model loaded for language={language!r} "
                f"({model_name}); falling back to default")
            model_name = self._default_model
            if model_name not in self._voices:
                self._load_model(model_name)
        return self._voices.get(model_name)

    # --- Pronunciation ---

    def _apply_pronunciations(self, text: str, language: Optional[str]) -> str:
        """Replace words with phonetic spellings for better TTS output."""
        if language and language in self._lang_pron:
            for word, replacement in self._lang_pron[language].items():
                text = re.sub(rf"\b{re.escape(word)}\b", replacement,
                              text, flags=re.IGNORECASE)
        return text

    # --- Core ---

    @property
    def interrupted(self) -> bool:
        """True if the last speak() was cut short by stop_speaking()."""
        return self._interrupted

    def stop_speaking(self):
        """Interrupt the current TTS playback. Safe to call from any thread."""
        self._stop_event.set()

    def speak(self, text: str, language: Optional[str] = None):
        """Blocking TTS: synthesize and play. Thread-safe.

        *language* selects the voice model (e.g. ``"fr"``, ``"de"``).
        Falls back to the default (English) if the language is unknown.
        Can be interrupted mid-playback via ``stop_speaking()``.
        After returning, check ``interrupted`` to know if it was cut short.
        """
        if not self._lock.acquire(timeout=0.5):
            self._emit(TtsEventType.TTS_BUSY, TtsBusyPayload(text))
            logger.warning(f"TTS busy, rejected: {text}")
            return

        try:
            voice = self._get_voice(language)
            if voice is None:
                logger.warning(f"TTS not ready, would say: {text}")
                return
            self._speaking = True
            self._interrupted = False
            self._stop_event.clear()
            tts_text = self._apply_pronunciations(text, language)
            start = time.time()
            self._emit(TtsEventType.TTS_STARTED,
                       TtsStartedPayload(text, tts_text if tts_text != text else None))
            self._play(tts_text, voice)
            duration = time.time() - start
            self._emit(TtsEventType.TTS_FINISHED,
                       TtsFinishedPayload(text, duration))
        except Exception as e:
            logger.error(f"TTS playback failed: {e}")
        finally:
            self._speaking = False
            self._lock.release()

    def speak_async(self, text: str, language: Optional[str] = None):
        """Non-blocking TTS: speak in a background thread."""
        threading.Thread(target=self.speak, args=(text, language),
                         daemon=True).start()

    def _play(self, text: str, voice: PiperVoice):
        sr = voice.config.sample_rate
        buffer = collections.deque()
        finished = threading.Event()
        stop = self._stop_event

        def callback(outdata, frames, time_info, status):
            if stop.is_set():
                outdata[:, 0] = 0
                raise sd.CallbackStop
            if buffer:
                chunk = buffer.popleft()
                if len(chunk) < frames:
                    outdata[:len(chunk), 0] = chunk
                    outdata[len(chunk):, 0] = 0
                else:
                    outdata[:, 0] = chunk[:frames]
                    if len(chunk) > frames:
                        buffer.appendleft(chunk[frames:])
            else:
                outdata[:, 0] = 0
                if finished.is_set():
                    raise sd.CallbackStop

        stream = sd.OutputStream(
            samplerate=sr, channels=1, dtype="float32",
            blocksize=4096, callback=callback,
        )
        stream.start()
        for audio_chunk in voice.synthesize(text):
            if stop.is_set():
                break
            buffer.append(audio_chunk.audio_float_array)
        finished.set()
        while stream.active:
            if stop.is_set():
                stream.abort()
                break
            sd.sleep(50)
        stream.close()
        if stop.is_set():
            self._interrupted = True
            logger.info("[TTS] Playback interrupted by stop_speaking()")

    def _emit(self, etype, payload):
        event = TtsEvent(type=etype, timestamp=time.time(), payload=payload)
        logger.info(f"[{etype.name}] {payload}")
        self._dispatcher.dispatch(event)


# ---------------------------------------------------------------------------
# Standalone mode
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Standalone voice output (TTS)")
    parser.add_argument("text", nargs="?", help="Text to speak (omit for interactive)")
    parser.add_argument("--language", "-l", default=None,
                        choices=sorted(LANGUAGE_MODELS.keys()),
                        help=f"Language code (one of: {', '.join(sorted(LANGUAGE_MODELS.keys()))})")
    parser.add_argument("--model-dir", default=PIPER_MODEL_DIR)
    parser.add_argument("--model-name", default=None,
                        help="Override Piper model name (default: from --language)")
    parser.add_argument("--no-pronunciations", "-P", action="store_true",
                        help="Disable pronunciation replacements")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode: type text, press Enter to speak")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    # Validate language
    if args.language and args.language not in LANGUAGE_MODELS:
        print(f"Unknown language: {args.language}  "
              f"(available: {', '.join(sorted(LANGUAGE_MODELS))})")
        return

    # Resolve model name: explicit --model-name > language lookup > default
    model_name = args.model_name
    if model_name is None:
        model_name = LANGUAGE_MODELS.get(args.language, PIPER_MODEL_NAME)

    tts = VoiceOutput(model_dir=args.model_dir, model_name=model_name)
    if args.no_pronunciations:
        tts._lang_pron = {}

    def on_event(event: TtsEvent):
        ts = datetime.now().strftime("%H:%M:%S")
        if event.type == TtsEventType.TTS_STARTED:
            print(f"  [{ts}] Speaking: \"{event.payload.text}\"")
            if event.payload.tts_text:
                print(f"  [{ts}] Pronounce: \"{event.payload.tts_text}\"")
        elif event.type == TtsEventType.TTS_FINISHED:
            print(f"  [{ts}] Done ({event.payload.duration_s:.1f}s)")
        elif event.type == TtsEventType.MODEL_READY:
            print(f"  [{ts}] Model loaded: {event.payload.model_name}")

    tts.subscribe(on_event)

    print("Loading TTS model...")
    tts.load_sync()
    if not tts.ready:
        print("Failed to load TTS model.")
        return

    language = args.language

    if args.text and not args.interactive:
        tts.speak(args.text, language=language)
    else:
        print("Type text and press Enter to speak. Ctrl+C to quit.\n")
        try:
            while True:
                text = input("> ")
                if text.strip():
                    tts.speak(text.strip(), language=language)
        except (KeyboardInterrupt, EOFError):
            pass

    print("\nDone.")


if __name__ == "__main__":
    main()
