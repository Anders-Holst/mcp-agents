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

import os
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

PIPER_MODEL_DIR = "piper_models"
PIPER_MODEL_NAME = "en_US-lessac-medium"


def _piper_download_url(model_name: str) -> str:
    """Derive the HuggingFace download URL from a piper model name.
    E.g. 'sv_SE-nst-medium' -> .../sv/sv_SE/nst/medium/sv_SE-nst-medium.onnx
    """
    parts = model_name.split("-")  # ['sv_SE', 'nst', 'medium']
    lang_region = parts[0]         # 'sv_SE'
    lang = lang_region.split("_")[0]  # 'sv'
    speaker = parts[1]             # 'nst'
    quality = parts[2] if len(parts) > 2 else "medium"
    base = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    return f"{base}/{lang}/{lang_region}/{speaker}/{quality}/"


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
                 model_name: str = PIPER_MODEL_NAME):
        self._model_dir = model_dir
        self._model_name = model_name
        self._piper_voice = None
        self._ready = False
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
    def speaking(self) -> bool:
        return self._speaking

    # --- Lifecycle ---

    def load(self):
        """Load TTS model in background thread."""
        threading.Thread(target=self._load, daemon=True).start()

    def load_sync(self):
        """Load TTS model synchronously (blocking)."""
        self._load()

    def _load(self):
        self._emit(TtsEventType.MODEL_LOADING,
                   ModelLoadingPayload(self._model_name))
        try:
            self._ensure_model()
            model_path = os.path.join(self._model_dir, f"{self._model_name}.onnx")
            logger.info(f"Loading piper model: {self._model_name}...")
            self._piper_voice = PiperVoice.load(model_path)
            self._ready = True
            self._emit(TtsEventType.MODEL_READY,
                       ModelReadyPayload(self._model_name))
            logger.info("Piper TTS loaded.")
        except Exception as e:
            self._emit(TtsEventType.MODEL_LOAD_FAILED,
                       ModelLoadFailedPayload(self._model_name, str(e)))
            logger.error(f"Failed to load piper: {e}")

    def _ensure_model(self):
        os.makedirs(self._model_dir, exist_ok=True)
        model_path = os.path.join(self._model_dir, f"{self._model_name}.onnx")
        config_path = model_path + ".json"
        if not os.path.exists(model_path):
            import urllib.request
            base_url = _piper_download_url(self._model_name)
            logger.info(f"Downloading piper model: {self._model_name}...")
            urllib.request.urlretrieve(
                base_url + f"{self._model_name}.onnx", model_path)
            urllib.request.urlretrieve(
                base_url + f"{self._model_name}.onnx.json", config_path)
            logger.info("Piper model downloaded.")

    # --- Core ---

    @property
    def interrupted(self) -> bool:
        """True if the last speak() was cut short by stop_speaking()."""
        return self._interrupted

    def stop_speaking(self):
        """Interrupt the current TTS playback. Safe to call from any thread."""
        self._stop_event.set()

    def speak(self, text: str):
        """Blocking TTS: synthesize and play. Thread-safe.

        Can be interrupted mid-playback via ``stop_speaking()``.
        After returning, check ``interrupted`` to know if it was cut short.
        """
        if not self._ready:
            logger.warning(f"TTS not ready, would say: {text}")
            return

        if not self._lock.acquire(timeout=0.5):
            self._emit(TtsEventType.TTS_BUSY, TtsBusyPayload(text))
            logger.warning(f"TTS busy, rejected: {text}")
            return

        try:
            self._speaking = True
            self._interrupted = False
            self._stop_event.clear()
            start = time.time()
            self._emit(TtsEventType.TTS_STARTED, TtsStartedPayload(text))
            self._play(text)
            duration = time.time() - start
            self._emit(TtsEventType.TTS_FINISHED,
                       TtsFinishedPayload(text, duration))
        finally:
            self._speaking = False
            self._lock.release()

    def speak_async(self, text: str):
        """Non-blocking TTS: speak in a background thread."""
        threading.Thread(target=self.speak, args=(text,), daemon=True).start()

    def _play(self, text: str):
        sr = self._piper_voice.config.sample_rate
        buffer = []
        finished = threading.Event()
        stop = self._stop_event

        def callback(outdata, frames, time_info, status):
            if stop.is_set():
                outdata[:, 0] = 0
                raise sd.CallbackStop
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
        for audio_chunk in self._piper_voice.synthesize(text):
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
    parser.add_argument("--model-dir", default=PIPER_MODEL_DIR)
    parser.add_argument("--model-name", default=PIPER_MODEL_NAME)
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode: type text, press Enter to speak")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    tts = VoiceOutput(model_dir=args.model_dir, model_name=args.model_name)

    def on_event(event: TtsEvent):
        ts = datetime.now().strftime("%H:%M:%S")
        if event.type == TtsEventType.TTS_STARTED:
            print(f"  [{ts}] Speaking: \"{event.payload.text}\"")
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

    if args.text and not args.interactive:
        tts.speak(args.text)
    else:
        print("Type text and press Enter to speak. Ctrl+C to quit.\n")
        try:
            while True:
                text = input("> ")
                if text.strip():
                    tts.speak(text.strip())
        except (KeyboardInterrupt, EOFError):
            pass

    print("\nDone.")


if __name__ == "__main__":
    main()
