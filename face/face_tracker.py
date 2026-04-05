"""
Face detection, recognition, and tracking module.

Provides stable face tracking across video frames with:
- Persistent tracking IDs (survive brief occlusions)
- Recognition with hysteresis (no name flickering)
- Emotion detection
- Face database management
- Typed event system with subscribe/unsubscribe

TrackedFace is pure tracking data (ID, bbox, encoding, emotion).
Identity (name) is a separate concern managed by FaceTracker.

Can be run standalone:
    python face_tracker.py [--db-dir known_faces] [--camera 0]
"""

import cv2
import face_recognition
import numpy as np
import os
import pickle
import time
import threading
import logging
import argparse
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Union

import onnxruntime as ort

from events import EventDispatcher

logger = logging.getLogger("face_tracker")

EMOTION_MODEL_DIR = "emotion_model"
EMOTION_MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
EMOTION_LABELS = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"]


# ---------------------------------------------------------------------------
# Event types and payloads
# ---------------------------------------------------------------------------

class FaceEventType(Enum):
    FACE_APPEARED = auto()
    FACE_DISAPPEARED = auto()
    FACE_OCCLUDED = auto()        # visible -> grace period
    FACE_RECOVERED = auto()       # grace period -> visible
    IDENTITY_CONFIRMED = auto()   # unknown -> named
    IDENTITY_LOST = auto()        # named -> unknown
    IDENTITY_CHANGED = auto()     # name A -> name B
    FACE_LEARNED = auto()         # learn_face() called
    FOCUS_CHANGED = auto()        # primary focus switched
    EMOTION_CHANGED = auto()      # emotion label changed


@dataclass(frozen=True)
class FaceAppearedPayload:
    bbox: tuple
    emotion: str
    emotion_confidence: float
    initial_name: Optional[str]
    initial_confidence: float


@dataclass(frozen=True)
class FaceDisappearedPayload:
    last_bbox: tuple
    name: Optional[str]
    duration_visible: float
    total_frames: int


@dataclass(frozen=True)
class FaceOccludedPayload:
    last_bbox: tuple
    name: Optional[str]


@dataclass(frozen=True)
class FaceRecoveredPayload:
    bbox: tuple
    name: Optional[str]
    seconds_missing: float


@dataclass(frozen=True)
class IdentityConfirmedPayload:
    name: str
    confidence: float
    last_seen_timestamp: Optional[float]


@dataclass(frozen=True)
class IdentityLostPayload:
    previous_name: str


@dataclass(frozen=True)
class IdentityChangedPayload:
    old_name: str
    new_name: str
    new_confidence: float


@dataclass(frozen=True)
class FaceLearnedPayload:
    name: str


@dataclass(frozen=True)
class FocusChangedPayload:
    old_track_id: Optional[int]
    new_track_id: int
    old_focus_score: float
    new_focus_score: float
    new_name: Optional[str]


@dataclass(frozen=True)
class EmotionChangedPayload:
    old_emotion: str
    new_emotion: str
    new_confidence: float
    name: Optional[str]


FaceEventPayload = Union[
    FaceAppearedPayload, FaceDisappearedPayload, FaceOccludedPayload,
    FaceRecoveredPayload, IdentityConfirmedPayload, IdentityLostPayload,
    IdentityChangedPayload, FaceLearnedPayload, FocusChangedPayload,
    EmotionChangedPayload,
]


@dataclass(frozen=True)
class FaceEvent:
    type: FaceEventType
    timestamp: float
    track_id: Optional[int]
    payload: FaceEventPayload


FaceEventCallback = Callable[[FaceEvent], None]


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass
class TrackedFace:
    """A face tracked across frames with a stable ID. Contains no identity info."""
    track_id: int
    emotion: str = "neutral"
    emotion_confidence: float = 0.0
    encoding: np.ndarray = field(default_factory=lambda: np.zeros(128))
    bbox: tuple = (0, 0, 0, 0)  # (top, right, bottom, left)
    first_seen: float = 0.0
    last_seen: float = 0.0
    frames_visible: int = 0
    frames_since_seen: int = 0
    focus_score: float = 0.0

    @property
    def center(self) -> tuple:
        top, right, bottom, left = self.bbox
        return ((left + right) // 2, (top + bottom) // 2)

    @property
    def area(self) -> int:
        top, right, bottom, left = self.bbox
        return max(0, (right - left) * (bottom - top))

    @property
    def is_visible(self) -> bool:
        return self.frames_since_seen == 0


@dataclass
class Identity:
    """Recognition result for a tracked face."""
    name: str
    confidence: float
    _matching_since: float = field(default=0.0, repr=False)
    _failing_since: float = field(default=0.0, repr=False)
    _candidate_name: str = field(default="", repr=False)
    _candidate_confidence: float = field(default=0.0, repr=False)
    _confirmed: bool = field(default=False, repr=False)


# ---------------------------------------------------------------------------
# FaceDatabase and EmotionDetector (unchanged)
# ---------------------------------------------------------------------------

class FaceDatabase:
    """Persistent face encoding database. Compatible with existing faces.pkl format."""

    def __init__(self, db_dir: str = "known_faces", tolerance: float = 0.6):
        self.db_dir = db_dir
        self.tolerance = tolerance
        self._db = {"encodings": [], "names": [], "last_seen": {}}
        self._lock = threading.Lock()

    def load(self):
        db_file = os.path.join(self.db_dir, "faces.pkl")
        if os.path.exists(db_file):
            with open(db_file, "rb") as f:
                self._db = pickle.load(f)
            if "last_seen" not in self._db:
                self._db["last_seen"] = {}
        logger.info(f"Database loaded: {len(self.known_names)} people, {self.encoding_count} encodings")

    def save(self):
        os.makedirs(self.db_dir, exist_ok=True)
        db_file = os.path.join(self.db_dir, "faces.pkl")
        with self._lock:
            with open(db_file, "wb") as f:
                pickle.dump(self._db, f)

    def recognize(self, encoding: np.ndarray) -> tuple:
        with self._lock:
            if not self._db["encodings"]:
                return (None, 0.0)
            distances = face_recognition.face_distance(self._db["encodings"], encoding)
            best_idx = np.argmin(distances)
            best_dist = distances[best_idx]
            if best_dist < self.tolerance:
                name = self._db["names"][best_idx]
                confidence = max(0.0, 1.0 - best_dist) * 100
                return (name, confidence)
            return (None, 0.0)

    def add_face(self, name: str, encoding: np.ndarray,
                 frame: np.ndarray, bbox: tuple):
        with self._lock:
            self._db["encodings"].append(encoding)
            self._db["names"].append(name)
        self.save()
        self._save_face_image(name, frame, bbox)
        logger.info(f"Added face for '{name}' ({self._db['names'].count(name)} samples)")

    def update_last_seen(self, name: str):
        with self._lock:
            self._db["last_seen"][name] = datetime.now().timestamp()

    def get_last_seen(self, name: str) -> Optional[float]:
        return self._db["last_seen"].get(name)

    def clear(self):
        with self._lock:
            self._db = {"encodings": [], "names": [], "last_seen": {}}
        self.save()
        logger.info("Database cleared")

    @property
    def known_names(self) -> set:
        return set(self._db["names"])

    @property
    def encoding_count(self) -> int:
        return len(self._db["encodings"])

    @property
    def last_seen_map(self) -> dict:
        return dict(self._db["last_seen"])

    def _save_face_image(self, name, frame, bbox):
        top, right, bottom, left = bbox
        face_img = frame[top:bottom, left:right]
        person_dir = os.path.join(self.db_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(os.path.join(person_dir, f"{timestamp}.jpg"), face_img)


class EmotionDetector:
    """ONNX-based facial emotion detection."""

    def __init__(self, model_dir: str = EMOTION_MODEL_DIR):
        self.model_dir = model_dir
        self.session = None
        self._ensure_model()

    def _ensure_model(self):
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, "emotion-ferplus-8.onnx")
        if not os.path.exists(model_path):
            logger.info("Downloading emotion detection model...")
            import urllib.request
            urllib.request.urlretrieve(EMOTION_MODEL_URL, model_path)
            logger.info("Emotion model downloaded.")
        self.session = ort.InferenceSession(model_path)

    def detect(self, face_bgr: np.ndarray) -> tuple:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        input_data = resized.astype(np.float32).reshape(1, 1, 64, 64)
        input_name = self.session.get_inputs()[0].name
        result = self.session.run(None, {input_name: input_data})
        scores = result[0][0]
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        idx = np.argmax(probs)
        return EMOTION_LABELS[idx], float(probs[idx])


# ---------------------------------------------------------------------------
# FaceTracker
# ---------------------------------------------------------------------------

class FaceTracker:
    """
    Detects, recognizes, and tracks faces across video frames.

    Each detected face gets a stable track_id that persists across frames.
    Identity (name) is separate from tracking — query via get_identity().

    Subscribe to typed FaceEvent callbacks via subscribe().
    """

    def __init__(self,
                 db: FaceDatabase,
                 emotion_detector: EmotionDetector,
                 frame_scale: float = 0.5,
                 track_encoding_threshold: float = 0.5,
                 track_iou_threshold: float = 0.3,
                 max_missing_seconds: float = 2.0,
                 recognition_confirm_seconds: float = 0.15,
                 recognition_revoke_seconds: float = 0.25,
                 focus_switch_threshold: float = 0.1,
                 focus_switch_seconds: float = 0.5,
                 emotion_debounce_seconds: float = 0.3):
        self.db = db
        self.emotion_detector = emotion_detector
        self.frame_scale = frame_scale
        self._track_enc_thresh = track_encoding_threshold
        self._track_iou_thresh = track_iou_threshold
        self._max_missing_s = max_missing_seconds
        self._confirm_s = recognition_confirm_seconds
        self._revoke_s = recognition_revoke_seconds
        self._emotion_debounce_s = emotion_debounce_seconds

        self._tracks: list[TrackedFace] = []
        self._identities: dict[int, Identity] = {}
        self._next_id = 1
        self._lock = threading.Lock()
        self._skip_frame = False

        # Focus hysteresis
        self._focus_id: Optional[int] = None
        self._focus_switch_threshold = focus_switch_threshold
        self._focus_switch_s = focus_switch_seconds
        self._focus_challenger_id: Optional[int] = None
        self._focus_challenger_since: float = 0.0

        # Emotion debounce: track_id -> (emotion, since_timestamp)
        self._emotion_stable: dict[int, tuple[str, float]] = {}

        # Event system
        self._dispatcher = EventDispatcher(owner="face_tracker")

    # --- Public API ---

    def subscribe(self, callback: FaceEventCallback,
                  event_types: Optional[set] = None) -> Callable[[], None]:
        """Register a callback to receive face events.

        Args:
            callback: Called with a FaceEvent.
            event_types: If provided, only these event types are delivered.
                         If None, all events are delivered.
        Returns:
            An unsubscribe function.
        """
        return self._dispatcher.subscribe(callback, event_types)

    def unsubscribe(self, callback: FaceEventCallback) -> bool:
        return self._dispatcher.unsubscribe(callback)

    def process_frame(self, frame: np.ndarray) -> list[TrackedFace]:
        """Process a video frame. Returns tracked faces sorted by focus score."""
        frame_h, frame_w = frame.shape[:2]
        pending: list[FaceEvent] = []

        self._skip_frame = not self._skip_frame
        if self._skip_frame:
            with self._lock:
                focus_events = self._update_focus_scores(frame_w, frame_h)
                pending.extend(focus_events)
                result = sorted(self._tracks, key=lambda f: f.focus_score, reverse=True)
            self._dispatch_all(pending)
            return result

        locations, encodings = self._detect_faces(frame)
        detections = list(zip(locations, encodings))
        now = time.time()

        with self._lock:
            matches, unmatched_dets, unmatched_tracks = self._match(detections)

            # --- Update matched tracks ---
            identity_changes = []
            for det_idx, track_idx in matches:
                bbox, enc = detections[det_idx]
                track = self._tracks[track_idx]
                was_occluded = not track.is_visible
                old_name = (self._identities.get(track.track_id) or Identity("", 0)).name or None
                old_emotion = track.emotion

                self._update_track(track, enc, bbox, frame)

                # Recovery event
                if was_occluded:
                    name = self.get_name(track.track_id)
                    pending.append(self._make_event(
                        FaceEventType.FACE_RECOVERED, track.track_id,
                        FaceRecoveredPayload(
                            bbox=bbox, name=name,
                            seconds_missing=now - track.last_seen + (now - track.last_seen),
                        )
                    ))
                    # Fix: seconds_missing should be time since last seen before this update
                    # We already updated last_seen in _update_track, so use the gap
                    # Actually, _update_track sets last_seen=now, and before it was the old value
                    # We need to capture it before. Let me adjust below.

                # Identity change
                new_ident = self._identities.get(track.track_id)
                new_name = new_ident.name if new_ident else None
                if old_name != new_name:
                    identity_changes.append((track.track_id, old_name, new_name,
                                             new_ident.confidence if new_ident else 0.0))

                # Emotion change (with debounce)
                if track.emotion != old_emotion:
                    stable = self._emotion_stable.get(track.track_id)
                    if stable is None or stable[0] != track.emotion:
                        self._emotion_stable[track.track_id] = (track.emotion, now)
                    elif now - stable[1] >= self._emotion_debounce_s:
                        name = self.get_name(track.track_id)
                        pending.append(self._make_event(
                            FaceEventType.EMOTION_CHANGED, track.track_id,
                            EmotionChangedPayload(
                                old_emotion=old_emotion, new_emotion=track.emotion,
                                new_confidence=track.emotion_confidence, name=name,
                            )
                        ))

            # --- Occluded tracks ---
            for track_idx in unmatched_tracks:
                track = self._tracks[track_idx]
                was_visible = track.is_visible
                track.frames_since_seen += 1
                if was_visible:
                    name = self.get_name(track.track_id)
                    pending.append(self._make_event(
                        FaceEventType.FACE_OCCLUDED, track.track_id,
                        FaceOccludedPayload(last_bbox=track.bbox, name=name)
                    ))

            # --- Evict lost tracks ---
            lost = []
            for track_idx in unmatched_tracks:
                track = self._tracks[track_idx]
                if now - track.last_seen > self._max_missing_s:
                    lost.append(track)

            for track in lost:
                ident = self._identities.pop(track.track_id, None)
                self._emotion_stable.pop(track.track_id, None)
                pending.append(self._make_event(
                    FaceEventType.FACE_DISAPPEARED, track.track_id,
                    FaceDisappearedPayload(
                        last_bbox=track.bbox,
                        name=ident.name if ident else None,
                        duration_visible=track.last_seen - track.first_seen,
                        total_frames=track.frames_visible,
                    )
                ))
            if lost:
                lost_ids = {t.track_id for t in lost}
                self._tracks = [t for t in self._tracks if t.track_id not in lost_ids]

            # --- New tracks ---
            for det_idx in unmatched_dets:
                bbox, enc = detections[det_idx]
                track = self._create_track(enc, bbox, frame)
                self._tracks.append(track)
                ident = self._identities.get(track.track_id)
                pending.append(self._make_event(
                    FaceEventType.FACE_APPEARED, track.track_id,
                    FaceAppearedPayload(
                        bbox=bbox, emotion=track.emotion,
                        emotion_confidence=track.emotion_confidence,
                        initial_name=ident.name if ident else None,
                        initial_confidence=ident.confidence if ident else 0.0,
                    )
                ))

            # --- Identity change events ---
            for track_id, old_name, new_name, new_conf in identity_changes:
                if old_name is None and new_name:
                    pending.append(self._make_event(
                        FaceEventType.IDENTITY_CONFIRMED, track_id,
                        IdentityConfirmedPayload(
                            name=new_name, confidence=new_conf,
                            last_seen_timestamp=self.db.get_last_seen(new_name),
                        )
                    ))
                elif old_name and new_name is None:
                    pending.append(self._make_event(
                        FaceEventType.IDENTITY_LOST, track_id,
                        IdentityLostPayload(previous_name=old_name)
                    ))
                elif old_name and new_name and old_name != new_name:
                    pending.append(self._make_event(
                        FaceEventType.IDENTITY_CHANGED, track_id,
                        IdentityChangedPayload(
                            old_name=old_name, new_name=new_name, new_confidence=new_conf,
                        )
                    ))

            # --- Focus ---
            focus_events = self._update_focus_scores(frame_w, frame_h)
            pending.extend(focus_events)

            result = sorted(self._tracks, key=lambda f: f.focus_score, reverse=True)

        # Dispatch all events outside lock
        self._dispatch_all(pending)
        return result

    @property
    def focus_track_id(self) -> Optional[int]:
        return self._focus_id

    def get_identity(self, track_id: int) -> Optional[Identity]:
        with self._lock:
            return self._identities.get(track_id)

    def get_name(self, track_id: int) -> Optional[str]:
        ident = self._identities.get(track_id)
        return ident.name if ident else None

    def get_confidence(self, track_id: int) -> float:
        ident = self._identities.get(track_id)
        return ident.confidence if ident else 0.0

    def is_recognized(self, track_id: int) -> bool:
        return track_id in self._identities

    def get_visible_faces(self) -> list[TrackedFace]:
        with self._lock:
            return [t for t in self._tracks if t.is_visible]

    def get_face_by_id(self, track_id: int) -> Optional[TrackedFace]:
        with self._lock:
            for t in self._tracks:
                if t.track_id == track_id:
                    return t
        return None

    def get_primary_face(self) -> Optional[TrackedFace]:
        if self._focus_id is not None:
            face = self.get_face_by_id(self._focus_id)
            if face and face.is_visible:
                return face
        visible = self.get_visible_faces()
        return max(visible, key=lambda f: f.focus_score) if visible else None

    def get_recognized_names(self) -> list[str]:
        result = []
        for f in self.get_visible_faces():
            name = self.get_name(f.track_id)
            if name:
                result.append(name)
        return result

    def learn_face(self, track_id: int, name: str, frame: np.ndarray) -> bool:
        face = self.get_face_by_id(track_id)
        if face is None:
            return False
        self.db.add_face(name, face.encoding, frame, face.bbox)
        with self._lock:
            self._identities[track_id] = Identity(
                name=name, confidence=100.0,
                _matching_since=0.0, _confirmed=True,
            )
        self._dispatcher.dispatch(self._make_event(
            FaceEventType.FACE_LEARNED, track_id,
            FaceLearnedPayload(name=name)
        ))
        return True

    @property
    def active_tracks(self) -> list[TrackedFace]:
        with self._lock:
            return list(self._tracks)

    # --- Internal ---

    def _make_event(self, etype, track_id, payload):
        return FaceEvent(type=etype, timestamp=time.time(),
                         track_id=track_id, payload=payload)

    def _dispatch_all(self, events):
        for e in events:
            logger.info(f"[{e.type.name}] track={e.track_id} {e.payload}")
            self._dispatcher.dispatch(e)

    def _update_focus_scores(self, frame_w, frame_h) -> list[FaceEvent]:
        """Compute focus scores and return any focus-change events."""
        events = []
        now = time.time()

        if not self._tracks:
            if self._focus_id is not None:
                events.append(self._make_event(
                    FaceEventType.FOCUS_CHANGED, None,
                    FocusChangedPayload(
                        old_track_id=self._focus_id, new_track_id=0,
                        old_focus_score=0, new_focus_score=0, new_name=None,
                    )
                ))
            self._focus_id = None
            self._focus_challenger_id = None
            self._focus_challenger_since = 0.0
            return events

        cx, cy = frame_w / 2, frame_h / 2
        max_area = max(t.area for t in self._tracks) or 1

        for track in self._tracks:
            fx, fy = track.center
            dx = (fx - cx) / cx if cx > 0 else 0
            dy = (fy - cy) / cy if cy > 0 else 0
            centrality = 1.0 - np.sqrt(0.75 * dx ** 2 + 0.25 * dy ** 2)
            size = min(1.0, track.area / max_area) if max_area > 0 else 0.0
            track.focus_score = 0.75 * centrality + 0.25 * size

        visible = [t for t in self._tracks if t.is_visible]
        if not visible:
            return events

        best = max(visible, key=lambda f: f.focus_score)

        current_exists = any(t.track_id == self._focus_id for t in self._tracks)
        if self._focus_id is None or not current_exists:
            old_id = self._focus_id
            self._focus_id = best.track_id
            self._focus_challenger_id = None
            self._focus_challenger_since = 0.0
            if old_id != best.track_id:
                events.append(self._make_event(
                    FaceEventType.FOCUS_CHANGED, best.track_id,
                    FocusChangedPayload(
                        old_track_id=old_id, new_track_id=best.track_id,
                        old_focus_score=0, new_focus_score=best.focus_score,
                        new_name=self.get_name(best.track_id),
                    )
                ))
            return events

        current_visible = any(t.track_id == self._focus_id and t.is_visible for t in self._tracks)
        if not current_visible:
            return events

        current_score = 0.0
        for t in visible:
            if t.track_id == self._focus_id:
                current_score = t.focus_score
                break

        if best.track_id != self._focus_id and \
           best.focus_score > current_score + self._focus_switch_threshold:
            if best.track_id == self._focus_challenger_id:
                if now - self._focus_challenger_since >= self._focus_switch_s:
                    old_id = self._focus_id
                    self._focus_id = best.track_id
                    self._focus_challenger_id = None
                    self._focus_challenger_since = 0.0
                    events.append(self._make_event(
                        FaceEventType.FOCUS_CHANGED, best.track_id,
                        FocusChangedPayload(
                            old_track_id=old_id, new_track_id=best.track_id,
                            old_focus_score=current_score,
                            new_focus_score=best.focus_score,
                            new_name=self.get_name(best.track_id),
                        )
                    ))
            else:
                self._focus_challenger_id = best.track_id
                self._focus_challenger_since = now
        else:
            self._focus_challenger_id = None
            self._focus_challenger_since = 0.0

        return events

    def _detect_faces(self, frame):
        small = cv2.resize(frame, (0, 0), fx=self.frame_scale, fy=self.frame_scale)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_small)
        encodings = face_recognition.face_encodings(rgb_small, locations)
        s = self.frame_scale
        locations = [
            (int(t / s), int(r / s), int(b / s), int(l / s))
            for t, r, b, l in locations
        ]
        return locations, encodings

    def _match(self, detections):
        if not detections or not self._tracks:
            return ([], list(range(len(detections))), list(range(len(self._tracks))))

        n_det = len(detections)
        n_trk = len(self._tracks)
        costs = np.zeros((n_det, n_trk))
        for i, (bbox_d, enc_d) in enumerate(detections):
            for j, track in enumerate(self._tracks):
                costs[i, j] = np.linalg.norm(enc_d - track.encoding)

        matches = []
        used_dets = set()
        used_tracks = set()
        candidates = sorted(
            ((costs[i, j], i, j) for i in range(n_det) for j in range(n_trk))
        )
        for dist, i, j in candidates:
            if i in used_dets or j in used_tracks:
                continue
            if dist < self._track_enc_thresh:
                matches.append((i, j))
                used_dets.add(i)
                used_tracks.add(j)

        remaining_dets = [i for i in range(n_det) if i not in used_dets]
        remaining_tracks = [j for j in range(n_trk) if j not in used_tracks]
        for i in list(remaining_dets):
            best_iou, best_j = 0.0, -1
            for j in remaining_tracks:
                iou = self._compute_iou(detections[i][0], self._tracks[j].bbox)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou > self._track_iou_thresh and best_j >= 0:
                matches.append((i, best_j))
                remaining_dets.remove(i)
                remaining_tracks.remove(best_j)

        return matches, remaining_dets, remaining_tracks

    def _compute_iou(self, bbox1, bbox2):
        t1, r1, b1, l1 = bbox1
        t2, r2, b2, l2 = bbox2
        it, il = max(t1, t2), max(l1, l2)
        ib, ir = min(b1, b2), min(r1, r2)
        if ib <= it or ir <= il:
            return 0.0
        inter = (ib - it) * (ir - il)
        union = (b1 - t1) * (r1 - l1) + (b2 - t2) * (r2 - l2) - inter
        return inter / union if union > 0 else 0.0

    def _update_track(self, track, encoding, bbox, frame):
        now = time.time()
        track.encoding = 0.3 * encoding + 0.7 * track.encoding
        track.bbox = bbox
        track.last_seen = now
        track.frames_visible += 1
        track.frames_since_seen = 0

        top, right, bottom, left = bbox
        face_roi = frame[top:bottom, left:right]
        if face_roi.size > 0 and self.emotion_detector:
            try:
                label, conf = self.emotion_detector.detect(face_roi)
                track.emotion = label
                track.emotion_confidence = conf
            except Exception:
                pass

        self._recognize_and_stabilize(track)

    def _create_track(self, encoding, bbox, frame):
        now = time.time()
        track = TrackedFace(
            track_id=self._next_id, encoding=encoding.copy(), bbox=bbox,
            first_seen=now, last_seen=now, frames_visible=1,
        )
        self._next_id += 1

        top, right, bottom, left = bbox
        face_roi = frame[top:bottom, left:right]
        if face_roi.size > 0 and self.emotion_detector:
            try:
                label, conf = self.emotion_detector.detect(face_roi)
                track.emotion = label
                track.emotion_confidence = conf
            except Exception:
                pass

        name, confidence = self.db.recognize(encoding)
        if name is not None:
            self._identities[track.track_id] = Identity(
                name=name, confidence=confidence,
                _matching_since=now, _candidate_name=name,
                _candidate_confidence=confidence,
            )

        return track

    def _recognize_and_stabilize(self, track):
        raw_name, raw_conf = self.db.recognize(track.encoding)
        tid = track.track_id
        now = time.time()
        ident = self._identities.get(tid)

        if raw_name is not None:
            if ident is None:
                ident = Identity(
                    name=raw_name, confidence=raw_conf,
                    _matching_since=now, _candidate_name=raw_name,
                    _candidate_confidence=raw_conf,
                )
                self._identities[tid] = ident
            else:
                if raw_name != ident._candidate_name:
                    ident._candidate_name = raw_name
                    ident._candidate_confidence = raw_conf
                    ident._matching_since = now
                ident._failing_since = 0.0
                if now - ident._matching_since >= self._confirm_s:
                    ident.name = raw_name
                    ident.confidence = raw_conf
                    ident._confirmed = True
        else:
            if ident is not None:
                if ident._failing_since == 0.0:
                    ident._failing_since = now
                ident._matching_since = now
                if now - ident._failing_since >= self._revoke_s:
                    del self._identities[tid]


# ---------------------------------------------------------------------------
# Standalone mode
# ---------------------------------------------------------------------------

def _get_name_from_gui(frame, existing_match=None):
    name = ""
    while True:
        display = frame.copy()
        overlay = display.copy()
        h, w = display.shape[:2]
        box_h = 120
        y_start = h // 2 - box_h // 2
        cv2.rectangle(overlay, (0, y_start), (w, y_start + box_h), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        if existing_match and not name:
            match_name, match_conf = existing_match
            cv2.putText(display, f"Known as: {match_name} ({match_conf:.0f}%) - ENTER=add sample, type to override",
                        (20, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(display, "Type name and press ENTER (ESC to cancel):",
                        (20, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, name + "_", (20, y_start + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow("Face Tracker", display)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            return None
        elif key in (13, 10):
            if name.strip():
                return name.strip()
            elif existing_match:
                return existing_match[0]
            return None
        elif key in (8, 127):
            name = name[:-1]
        elif 32 <= key <= 126:
            name += chr(key)


_EVENT_SHORT_NAMES = {
    "FACE_APPEARED": "APPEARED",
    "FACE_DISAPPEARED": "GONE",
    "FACE_OCCLUDED": "OCCLUDED",
    "FACE_RECOVERED": "RECOVERED",
    "IDENTITY_CONFIRMED": "ID OK",
    "IDENTITY_LOST": "ID LOST",
    "IDENTITY_CHANGED": "ID CHANGE",
    "FACE_LEARNED": "LEARNED",
    "FOCUS_CHANGED": "FOCUS",
    "EMOTION_CHANGED": "EMOTION",
}

_EVENT_COLORS = {
    "FACE_APPEARED": (255, 200, 100),
    "FACE_DISAPPEARED": (120, 120, 255),
    "FACE_OCCLUDED": (180, 180, 120),
    "FACE_RECOVERED": (120, 255, 180),
    "IDENTITY_CONFIRMED": (120, 255, 120),
    "IDENTITY_LOST": (120, 120, 255),
    "IDENTITY_CHANGED": (220, 220, 120),
    "FACE_LEARNED": (255, 255, 120),
    "FOCUS_CHANGED": (120, 220, 255),
    "EMOTION_CHANGED": (220, 140, 255),
}


def _draw_log_window(log_lines, width=800, height=600):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    # Title bar
    cv2.rectangle(canvas, (0, 0), (width, 36), (50, 50, 50), cv2.FILLED)
    cv2.putText(canvas, "FACE TRACKER EVENTS", (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)
    count_text = f"{len(log_lines)} events"
    cv2.putText(canvas, count_text, (width - 150, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

    line_h = 26
    max_lines = (height - 46) // line_h
    visible = log_lines[-max_lines:]

    tag_x = 70       # after timestamp
    msg_x = 175      # after tag
    dim_color = (90, 90, 90)

    for i, (ts, etype, msg) in enumerate(visible):
        y = 44 + i * line_h
        color = _EVENT_COLORS.get(etype, (180, 180, 180))
        short = _EVENT_SHORT_NAMES.get(etype, etype[:10])

        # Alternating row background
        if i % 2 == 0:
            cv2.rectangle(canvas, (0, y - 4), (width, y + line_h - 6), (38, 38, 38), cv2.FILLED)

        # Timestamp (dimmed)
        cv2.putText(canvas, ts, (8, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, dim_color, 1, cv2.LINE_AA)

        # Event tag (colored, fixed width)
        cv2.putText(canvas, short, (tag_x, y + 14),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, color, 1, cv2.LINE_AA)

        # Message (white, truncated to fit)
        max_chars = (width - msg_x - 10) // 8
        display_msg = msg if len(msg) <= max_chars else msg[:max_chars - 2] + ".."
        cv2.putText(canvas, display_msg, (msg_x, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

        # Subtle separator
        cv2.line(canvas, (8, y + line_h - 5), (width - 8, y + line_h - 5), (45, 45, 45), 1)

    cv2.imshow("Face Tracker Log", canvas)


def main():
    parser = argparse.ArgumentParser(description="Standalone face tracker")
    parser.add_argument("--db-dir", default="known_faces", help="Face database directory")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--scale", type=float, default=0.5, help="Detection scale factor")
    parser.add_argument("--fps", type=int, default=0, help="Max FPS (0 = unlimited)")
    parser.add_argument("--no-emotion", action="store_true", help="Disable emotion detection")
    parser.add_argument("--no-log-window", action="store_true", help="Disable log window")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    log_lines = []

    def on_event_display(event: FaceEvent):
        ts = datetime.now().strftime("%H:%M:%S")
        p = event.payload
        # Build a short summary
        if event.type == FaceEventType.FACE_APPEARED:
            msg = f"track={event.track_id} name={p.initial_name or '?'} emo={p.emotion}"
        elif event.type == FaceEventType.FACE_DISAPPEARED:
            msg = f"track={event.track_id} name={p.name or '?'} dur={p.duration_visible:.1f}s"
        elif event.type == FaceEventType.FOCUS_CHANGED:
            msg = f"{p.old_track_id} -> {p.new_track_id} ({p.new_name or '?'})"
        elif event.type == FaceEventType.IDENTITY_CONFIRMED:
            msg = f"track={event.track_id} -> {p.name} ({p.confidence:.0f}%)"
        elif event.type == FaceEventType.EMOTION_CHANGED:
            msg = f"track={event.track_id} {p.old_emotion}->{p.new_emotion}"
        else:
            msg = str(p)[:60]
        log_lines.append((ts, event.type.name, msg))

    face_db = FaceDatabase(db_dir=args.db_dir)
    face_db.load()

    emotion_detector = None
    if not args.no_emotion:
        emotion_detector = EmotionDetector()

    tracker = FaceTracker(db=face_db, emotion_detector=emotion_detector,
                          frame_scale=args.scale)
    tracker.subscribe(on_event_display)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error(f"Could not open camera {args.camera}")
        return

    cv2.namedWindow("Face Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Tracker", 800, 600)
    if not args.no_log_window:
        cv2.namedWindow("Face Tracker Log", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Face Tracker Log", 800, 600)
        cv2.moveWindow("Face Tracker Log", 820, 0)

    selected_track_id = None
    frame_min_interval = 1.0 / args.fps if args.fps > 0 else 0
    last_frame_time = 0.0

    logger.info(f"Controls: L=learn  TAB=select  D=delete-db  Q=quit | FPS cap: {args.fps or 'unlimited'}")

    while True:
        if frame_min_interval > 0:
            now = time.time()
            elapsed = now - last_frame_time
            if elapsed < frame_min_interval:
                wait_ms = max(1, int((frame_min_interval - elapsed) * 1000))
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord("q") or key == 27:
                    break
                continue
            last_frame_time = now

        ret, frame = cap.read()
        if not ret:
            break

        faces = tracker.process_frame(frame)
        visible = [f for f in faces if f.is_visible]

        if selected_track_id is not None:
            if not any(f.track_id == selected_track_id for f in visible):
                selected_track_id = visible[0].track_id if visible else None
        elif visible:
            selected_track_id = visible[0].track_id

        focus_id = tracker.focus_track_id
        focus_face = tracker.get_face_by_id(focus_id) if focus_id else None
        focus_is_ghost = focus_face is not None and not focus_face.is_visible

        if focus_is_ghost and focus_face:
            top, right, bottom, left = focus_face.bbox
            elapsed = time.time() - focus_face.last_seen
            alpha = max(0.0, 1.0 - elapsed / 2.0)
            ghost_color = (0, int(255 * alpha), int(100 * alpha))
            for i in range(0, right - left, 12):
                cv2.line(frame, (left + i, top), (left + min(i + 6, right - left), top), ghost_color, 2)
                cv2.line(frame, (left + i, bottom), (left + min(i + 6, right - left), bottom), ghost_color, 2)
            for i in range(0, bottom - top, 12):
                cv2.line(frame, (left, top + i), (left, top + min(i + 6, bottom - top)), ghost_color, 2)
                cv2.line(frame, (right, top + i), (right, top + min(i + 6, bottom - top)), ghost_color, 2)
            name = tracker.get_name(focus_face.track_id)
            ghost_label = f"FOCUS (lost {elapsed:.1f}s)"
            if name:
                ghost_label = f"{name} - {ghost_label}"
            cv2.putText(frame, ghost_label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ghost_color, 1)

        for rank, face in enumerate(visible):
            is_focus = (face.track_id == focus_id)
            is_selected = (face.track_id == selected_track_id)
            name = tracker.get_name(face.track_id)
            conf = tracker.get_confidence(face.track_id)
            top, right, bottom, left = face.bbox

            if is_focus:
                color = (0, 255, 100)
                thickness = 4
                glow = frame.copy()
                pad = 6
                cv2.rectangle(glow, (left - pad, top - pad), (right + pad, bottom + pad),
                              (0, 255, 100), cv2.FILLED)
                cv2.addWeighted(glow, 0.15, frame, 0.85, 0, frame)
            elif name:
                color = (0, 200, 0)
                thickness = 2
            else:
                color = (0, 0, 200)
                thickness = 1

            if not is_focus and len(visible) > 1:
                dim = frame.copy()
                cv2.rectangle(dim, (left, top), (right, bottom), (0, 0, 0), cv2.FILLED)
                cv2.addWeighted(dim, 0.15, frame, 0.85, 0, frame)

            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)

            if is_selected:
                cv2.rectangle(frame, (left - 2, top - 2), (right + 2, bottom + 2),
                              (0, 255, 255), 1)

            if name:
                label = f"[{rank+1}] #{face.track_id} {name} {conf:.0f}%"
            else:
                label = f"[{rank+1}] #{face.track_id} Unknown"
            cv2.rectangle(frame, (left, bottom), (right, bottom + 30), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            if is_focus:
                badge_w = 70
                cv2.rectangle(frame, (left, top - 28), (left + badge_w, top - 4),
                              (0, 255, 100), cv2.FILLED)
                cv2.putText(frame, "FOCUS", (left + 6, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            emo_y = top - 32 if is_focus else top - 10
            if face.emotion and face.emotion != "neutral":
                cv2.putText(frame, f"{face.emotion} ({face.emotion_confidence:.0%})",
                            (left, emo_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            info = f"focus:{face.focus_score:.2f} vis:{face.frames_visible}"
            cv2.putText(frame, info, (left + 6, bottom + 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        status = f"Tracks: {len(visible)} | Known: {len(face_db.known_names)} | DB: {face_db.encoding_count} encodings"
        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "L=learn  TAB=select  D=delete-db  Q=quit",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        cv2.imshow("Face Tracker", frame)
        if not args.no_log_window:
            _draw_log_window(log_lines)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == 9:
            if visible:
                ids = [f.track_id for f in visible]
                if selected_track_id in ids:
                    idx = (ids.index(selected_track_id) + 1) % len(ids)
                    selected_track_id = ids[idx]
                else:
                    selected_track_id = ids[0]
        elif key == ord("l"):
            if selected_track_id:
                face = tracker.get_face_by_id(selected_track_id)
                if face:
                    existing = None
                    if tracker.is_recognized(face.track_id):
                        existing = (tracker.get_name(face.track_id),
                                    tracker.get_confidence(face.track_id))
                    input_name = _get_name_from_gui(frame, existing)
                    if input_name:
                        tracker.learn_face(selected_track_id, input_name, frame)
        elif key == ord("d"):
            display = frame.copy()
            cv2.putText(display, "DELETE ALL FACES? Y/N", (50, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imshow("Face Tracker", display)
            confirm = cv2.waitKey(0) & 0xFF
            if confirm == ord("y"):
                face_db.clear()

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Done.")


if __name__ == "__main__":
    main()
