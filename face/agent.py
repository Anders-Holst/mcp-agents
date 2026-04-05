"""
Agent module: the intelligence loop that ties perception to action.

Subscribes to face_tracker and voice_input events, queries people_memory,
decides what to do, and speaks via voice_output.

Emits its own AgentEvent for UI/logging.

Can be run standalone with face_tracker + voice_input + voice_output:
    python agent.py [--no-voice] [--no-auto-ask]
"""

import time
import threading
import logging
import argparse
from enum import Enum, auto
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Callable, Union

from events import EventDispatcher
from face_tracker import (
    FaceTracker, FaceDatabase, EmotionDetector, FaceEvent, FaceEventType,
)
from voice_input import VoiceInput, AudioMonitor, ContinuousListener
from voice_output import VoiceOutput
from people_memory import PeopleMemory
from llm import ConversationLLM

logger = logging.getLogger("agent")


# ---------------------------------------------------------------------------
# Agent event types
# ---------------------------------------------------------------------------

class AgentEventType(Enum):
    GREETING = auto()           # agent decided to greet someone
    GOODBYE = auto()            # agent said goodbye when someone left
    ASKING_NAME = auto()        # agent is asking an unknown face for their name
    RESPONDING = auto()         # agent is responding to speech
    LEARNED_NAME = auto()       # agent learned someone's name from speech
    NAME_EXTRACT_FAILED = auto()  # heard speech but couldn't extract a name
    THINKING = auto()           # agent is deciding what to do (reasoning)


@dataclass(frozen=True)
class GreetingPayload:
    track_id: int
    name: str
    text: str
    emotion: str


@dataclass(frozen=True)
class GoodbyePayload:
    track_id: int
    name: str
    text: str


@dataclass(frozen=True)
class AskingNamePayload:
    track_id: int
    text: str


@dataclass(frozen=True)
class RespondingPayload:
    track_id: Optional[int]
    name: Optional[str]
    heard: str
    response: str
    language: str


@dataclass(frozen=True)
class LearnedNamePayload:
    track_id: int
    name: str
    raw_speech: str


@dataclass(frozen=True)
class NameExtractFailedPayload:
    track_id: int
    raw_speech: str


@dataclass(frozen=True)
class ThinkingPayload:
    reason: str


AgentEventPayload = Union[
    GreetingPayload, GoodbyePayload, AskingNamePayload, RespondingPayload,
    LearnedNamePayload, NameExtractFailedPayload, ThinkingPayload,
]


@dataclass(frozen=True)
class AgentEvent:
    type: AgentEventType
    timestamp: float
    payload: AgentEventPayload


AgentEventCallback = Callable[[AgentEvent], None]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def extract_name(text: str) -> Optional[str]:
    """Try to extract a name from spoken text."""
    if not text:
        return None
    text = text.strip().strip(".")
    for prefix in ["my name is", "i'm", "i am", "they call me", "it's", "its",
                   "hi i'm", "hi i am", "hello i'm", "hello i am",
                   "hey i'm", "hey i am",
                   "jag heter", "mitt namn är"]:
        lower = text.lower()
        if lower.startswith(prefix):
            text = text[len(prefix):].strip().strip(".,!")
            break
    name = text.split()[0] if text else None
    if name:
        name = name.strip(".,!?").capitalize()
    return name if name and len(name) > 1 else None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """The intelligence loop: perceives, remembers, decides, acts.

    Subscribes to face and voice events, manages people memory,
    and speaks via voice output.
    """

    def __init__(self, *,
                 tracker: FaceTracker,
                 voice_input: VoiceInput,
                 voice_output: VoiceOutput,
                 memory: PeopleMemory,
                 llm: ConversationLLM,
                 greeting_cooldown_s: float = 60.0,
                 ask_name_cooldown_s: float = 30.0,
                 min_frames_before_ask: int = 3,
                 auto_ask: bool = True,
                 auto_greet: bool = True):
        self.tracker = tracker
        self.voice_in = voice_input
        self.voice_out = voice_output
        self.memory = memory
        self.llm = llm

        self._greeting_cooldown = greeting_cooldown_s
        self._ask_cooldown = ask_name_cooldown_s
        self._min_frames_ask = min_frames_before_ask
        self.auto_ask = auto_ask
        self.auto_greet = auto_greet

        self._busy = False
        self._busy_lock = threading.Lock()

        # Cooldown tracking
        self._greeted: dict[str, float] = {}      # name -> timestamp
        self._asked: dict[int, float] = {}         # track_id -> timestamp

        # Continuous listener
        self._listener: Optional[ContinuousListener] = None

        # Event system
        self._dispatcher = EventDispatcher(owner="agent")

        # Subscribe to face events
        self.tracker.subscribe(self._on_face_event)

    def subscribe(self, callback: AgentEventCallback,
                  event_types: Optional[set] = None) -> Callable[[], None]:
        return self._dispatcher.subscribe(callback, event_types)

    # --- Control ---

    def start(self):
        """Start continuous listening (paused until first interaction completes)."""
        self._listener = ContinuousListener(
            self.voice_in, on_heard=self._on_heard_speech)
        self._listener.start()
        self._listener.paused = True  # don't listen until first greet is done
        logger.info("Agent started (listening paused until first interaction)")

    def stop(self):
        """Stop continuous listening and clean up."""
        if self._listener:
            self._listener.stop()
            self._listener = None
        self.llm.stop()
        self.memory.save_all()
        logger.info("Agent stopped")

    @property
    def busy(self) -> bool:
        return self._busy

    def pause_listening(self):
        if self._listener:
            self._listener.paused = True

    def resume_listening(self):
        if self._listener:
            self._listener.paused = False

    # --- Manual actions ---

    def speak(self, text: str):
        """Speak text, pausing listener during playback."""
        self.pause_listening()
        self.voice_out.speak(text)
        self.resume_listening()

    def ask_name(self, track_id: int, frame=None):
        """Manually trigger asking an unknown face for their name."""
        threading.Thread(target=self._do_ask_name,
                         args=(track_id, frame), daemon=True).start()

    def greet(self, track_id: int):
        """Manually trigger greeting a known face."""
        threading.Thread(target=self._do_greet,
                         args=(track_id,), daemon=True).start()

    # --- Face event handler ---

    def _on_face_event(self, event: FaceEvent):
        if event.type in (FaceEventType.FACE_APPEARED, FaceEventType.IDENTITY_CONFIRMED):
            logger.debug(f"[AGENT] face event: {event.type.name} track={event.track_id} busy={self._busy}")

        # Goodbye is lightweight (no LLM) — handle even when busy
        if event.type == FaceEventType.FACE_DISAPPEARED:
            self._handle_face_disappeared(event)
            return

        if self._busy:
            if event.type in (FaceEventType.FACE_APPEARED, FaceEventType.IDENTITY_CONFIRMED):
                logger.info(f"[AGENT] SKIPPED {event.type.name} for track {event.track_id} — agent busy")
            return

        if event.type == FaceEventType.IDENTITY_CONFIRMED:
            self._handle_identity_confirmed(event)
        elif event.type == FaceEventType.FACE_APPEARED:
            self._handle_face_appeared(event)

    def _handle_identity_confirmed(self, event: FaceEvent):
        """A face was just identified — update memory and greet."""
        p = event.payload
        tid = event.track_id
        face = self.tracker.get_face_by_id(tid)
        self.memory.identify(tid, p.name)
        self.memory.update_seen(tid, face.emotion if face else "")
        self._try_greet(tid, p.name)

    def _handle_face_appeared(self, event: FaceEvent):
        """A new face appeared — track in memory, greet if known."""
        p = event.payload
        tid = event.track_id
        logger.info(f"[AGENT] FACE_APPEARED: track={tid} name={p.initial_name} emotion={p.emotion}")
        self.memory.get_or_create(tid)
        if p.initial_name:
            self.memory.identify(tid, p.initial_name)
            self.memory.update_seen(tid, p.emotion)
            self._try_greet(tid, p.initial_name)

    def _try_greet(self, track_id: int, name: str):
        """Greet a person if auto_greet is on, agent is free, and cooldown expired."""
        if not self.auto_greet or self._busy or not name:
            return
        now = time.time()
        if name in self._greeted and (now - self._greeted[name]) < self._greeting_cooldown:
            return
        logger.info(f"[AGENT] triggering greet for {name} (track {track_id})")
        self._greeted[name] = now
        threading.Thread(target=self._do_greet, args=(track_id,), daemon=True).start()

    def _handle_face_disappeared(self, event: FaceEvent):
        """A face left the frame — say goodbye if it was a known person."""
        p = event.payload
        name = p.name
        if not name:
            return

        tid = event.track_id
        logger.info(f"[AGENT] FACE_DISAPPEARED: track={tid} name={name} duration={p.duration_visible:.1f}s")

        text = f"Goodbye, {name}!"
        self._emit(AgentEventType.GOODBYE, GoodbyePayload(
            track_id=tid, name=name, text=text,
        ))
        self.speak(text)

    # --- Periodic check for unknown faces (called from outside or a timer) ---

    def check_unknown_faces(self, frame=None):
        """Check if any visible unknown faces should be asked for their name."""
        if not self.auto_ask or self._busy:
            return
        if not self.voice_in.ready or not self.voice_out.ready:
            return

        for face in self.tracker.get_visible_faces():
            if self.tracker.is_recognized(face.track_id):
                continue
            if face.frames_visible < self._min_frames_ask:
                continue
            tid = face.track_id
            now = time.time()
            if tid in self._asked and (now - self._asked[tid]) < self._ask_cooldown:
                continue

            self._asked[tid] = now
            threading.Thread(target=self._do_ask_name,
                             args=(tid, frame), daemon=True).start()
            break  # one at a time

    # --- Speech handler ---

    def _on_heard_speech(self, text: str):
        """Called by ContinuousListener when speech is transcribed."""
        if self._busy or not text:
            return

        with self._busy_lock:
            if self._busy:
                return
            self._busy = True

        try:
            self.pause_listening()

            # Who are we talking to?
            primary = self.tracker.get_primary_face()
            tid = primary.track_id if primary else None
            name = self.tracker.get_name(tid) if tid else None
            lang = self.voice_in.detected_language or "en"

            # Log dialogue
            if tid:
                self.memory.add_dialogue(tid, "person", text,
                                         language=lang,
                                         emotion=primary.emotion if primary else "")

            # Generate response via LLM
            response = self.llm.generate_response(self.memory, tid, text, language=lang)

            self._emit(AgentEventType.RESPONDING, RespondingPayload(
                track_id=tid, name=name, heard=text,
                response=response, language=lang,
            ))

            # Log and speak
            if tid:
                self.memory.add_dialogue(tid, "system", response, language=lang)
            self.speak(response)

        finally:
            self._busy = False
            self.resume_listening()

    # --- Internal action implementations ---

    def _do_greet(self, track_id: int):
        with self._busy_lock:
            if self._busy:
                logger.info(f"[AGENT] _do_greet skipped for track {track_id} — busy")
                return
            self._busy = True

        try:
            logger.info(f"[AGENT] _do_greet starting for track {track_id}")
            self.pause_listening()

            face = self.tracker.get_face_by_id(track_id)
            name = self.tracker.get_name(track_id)
            emotion = face.emotion if face else ""

            if not name:
                logger.info(f"[AGENT] _do_greet: no name for track {track_id}, aborting")
                return

            logger.info(f"[AGENT] _do_greet: calling LLM for {name} (emotion={emotion})")
            greeting = self.llm.generate_greeting(self.memory, track_id, emotion)
            logger.info(f"[AGENT] _do_greet: LLM returned, speaking greeting")

            self._emit(AgentEventType.GREETING, GreetingPayload(
                track_id=track_id, name=name, text=greeting, emotion=emotion,
            ))

            self.memory.add_dialogue(track_id, "system", greeting)
            self.memory.update_seen(track_id, emotion)
            self.speak(greeting)
            logger.info(f"[AGENT] _do_greet: done, resuming listener")

        finally:
            self._busy = False
            self.resume_listening()

    def _do_ask_name(self, track_id: int, frame=None):
        with self._busy_lock:
            if self._busy:
                return
            self._busy = True

        try:
            self.pause_listening()

            ask_text = self.llm.generate_ask_name(track_id)
            self._emit(AgentEventType.ASKING_NAME, AskingNamePayload(
                track_id=track_id, text=ask_text,
            ))

            self.memory.add_dialogue(track_id, "system", ask_text)
            self.speak(ask_text)

            # Listen for response
            response = self.voice_in.listen()
            if response:
                self.memory.add_dialogue(track_id, "person", response,
                                         language=self.voice_in.detected_language or "")

            extracted = extract_name(response)
            if extracted:
                self._emit(AgentEventType.LEARNED_NAME, LearnedNamePayload(
                    track_id=track_id, name=extracted, raw_speech=response,
                ))

                self.memory.identify(track_id, extracted)
                if frame is not None:
                    self.tracker.learn_face(track_id, extracted, frame)

                reply = f"Nice to meet you, {extracted}!"
                self.memory.add_dialogue(track_id, "system", reply)
                self.speak(reply)
            else:
                self._emit(AgentEventType.NAME_EXTRACT_FAILED,
                           NameExtractFailedPayload(
                               track_id=track_id, raw_speech=response or "",
                           ))
                reply = "Sorry, I didn't catch that."
                self.memory.add_dialogue(track_id, "system", reply)
                self.speak(reply)

        finally:
            self._busy = False
            self.resume_listening()

    def _emit(self, etype, payload):
        event = AgentEvent(type=etype, timestamp=time.time(), payload=payload)
        logger.info(f"[{etype.name}] {payload}")
        self._dispatcher.dispatch(event)


# ---------------------------------------------------------------------------
# Standalone mode
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Standalone agent")
    parser.add_argument("--db-dir", default="known_faces")
    parser.add_argument("--people-dir", default="people")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--no-auto-ask", action="store_true")
    parser.add_argument("--no-auto-greet", action="store_true")
    parser.add_argument("--llm-model", default="qwen3:8b", help="Ollama model name")
    parser.add_argument("--ollama-url", default="http://localhost:11434/v1")
    parser.add_argument("--en-voice", default="en_US-lessac-medium",
                        help="English TTS voice")
    parser.add_argument("--mcp-config", default=None,
                        help="Path to MCP servers JSON config (default: mcp_servers.json)")
    parser.add_argument("--mcp-server", action="append", default=[],
                        help="MCP server SSE URL (can be repeated)")
    parser.add_argument("--agent-name", default="Face Agent",
                        help="Name the agent uses to describe itself")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    # Initialize all components
    face_db = FaceDatabase(db_dir=args.db_dir)
    face_db.load()
    emotion_detector = EmotionDetector()
    tracker = FaceTracker(db=face_db, emotion_detector=emotion_detector)

    voice_in = VoiceInput()
    voice_in.load()

    voice_out = VoiceOutput(model_name=args.en_voice)
    voice_out.load()

    memory = PeopleMemory(storage_dir=args.people_dir)
    memory.load()

    from mcp_client import load_servers
    mcp_servers, mcp_descriptions = load_servers(
        config_path=args.mcp_config, server_urls=args.mcp_server)

    llm = ConversationLLM(
        model_name=args.llm_model,
        ollama_url=args.ollama_url,
        mcp_servers=mcp_servers,
        mcp_descriptions=mcp_descriptions,
        agent_name=args.agent_name,
    )
    logger.info(f"LLM: {args.llm_model} via {args.ollama_url}")

    monitor = AudioMonitor()
    monitor.start()

    agent = Agent(
        tracker=tracker,
        voice_input=voice_in,
        voice_output=voice_out,
        memory=memory,
        llm=llm,
        auto_ask=not args.no_auto_ask,
        auto_greet=not args.no_auto_greet,
    )

    # Log agent events
    def on_agent_event(event: AgentEvent):
        ts = datetime.now().strftime("%H:%M:%S")
        p = event.payload
        if event.type == AgentEventType.GREETING:
            print(f"  [{ts}] GREET {p.name}: \"{p.text}\"")
        elif event.type == AgentEventType.GOODBYE:
            print(f"  [{ts}] GOODBYE {p.name}: \"{p.text}\"")
        elif event.type == AgentEventType.ASKING_NAME:
            print(f"  [{ts}] ASK track {p.track_id}: \"{p.text}\"")
        elif event.type == AgentEventType.RESPONDING:
            print(f"  [{ts}] HEARD from {p.name or '?'}: \"{p.heard}\"")
            print(f"  [{ts}] REPLY: \"{p.response}\"")
        elif event.type == AgentEventType.LEARNED_NAME:
            print(f"  [{ts}] LEARNED: track {p.track_id} = {p.name}")
        elif event.type == AgentEventType.NAME_EXTRACT_FAILED:
            print(f"  [{ts}] NAME FAIL: \"{p.raw_speech}\"")

    agent.subscribe(on_agent_event)

    # Wait for models
    print("Loading models...")
    while not voice_in.ready:
        time.sleep(0.5)
    while not voice_out.ready:
        time.sleep(0.5)
    print("Models loaded. Starting agent.\n")

    agent.start()

    import cv2
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error(f"Could not open camera {args.camera}")
        return

    cv2.namedWindow("Agent", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Agent", 800, 600)

    frame_interval = 1.0 / args.fps if args.fps > 0 else 0
    last_frame = 0.0

    print(f"Running at {args.fps} FPS. Q to quit.\n")

    try:
        while True:
            if frame_interval > 0:
                now = time.time()
                if now - last_frame < frame_interval:
                    wait_ms = max(1, int((frame_interval - (now - last_frame)) * 1000))
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key == ord("q") or key == 27:
                        break
                    continue
                last_frame = now

            ret, frame = cap.read()
            if not ret:
                break

            faces = tracker.process_frame(frame)
            visible = [f for f in faces if f.is_visible]

            # Let agent check for unknown faces
            agent.check_unknown_faces(frame)

            # Draw faces
            from main import draw_faces
            draw_faces(frame, tracker)

            status = f"Faces: {len(visible)} | Known: {len(face_db.known_names)} | People: {memory.active_count}"
            if agent.busy:
                status += " | BUSY"
            cv2.putText(frame, status, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Agent", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        pass

    agent.stop()
    monitor.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()
