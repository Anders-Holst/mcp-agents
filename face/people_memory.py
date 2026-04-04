"""
People memory module: stores per-person data (dialogues, facts, preferences).

All lookups are by face track_id. A person may or may not have a name.
Persists as JSON files — one per person in a configurable directory.

Can be used standalone or as part of the larger system.
"""

import json
import os
import time
import threading
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger("people_memory")


@dataclass
class DialogueEntry:
    """A single exchange in a conversation."""
    timestamp: float          # time.time()
    speaker: str              # "system" or "person"
    text: str
    language: str = ""
    emotion: str = ""


@dataclass
class Person:
    """Everything we know about a person, keyed by track_id at runtime."""
    track_id: int                          # current session track_id
    name: Optional[str] = None             # None until identified
    first_met: float = 0.0
    last_seen: float = 0.0
    last_talked: float = 0.0
    times_seen: int = 0
    emotion: str = ""

    dialogues: list[DialogueEntry] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    preferences: dict = field(default_factory=dict)
    summary: str = ""

    # Persistent ID for linking across sessions (set from face_db name or generated)
    persistent_id: str = ""

    @property
    def is_identified(self) -> bool:
        return self.name is not None

    def to_dict(self) -> dict:
        return {
            "persistent_id": self.persistent_id,
            "name": self.name,
            "first_met": self.first_met,
            "last_seen": self.last_seen,
            "last_talked": self.last_talked,
            "times_seen": self.times_seen,
            "emotion": self.emotion,
            "facts": self.facts,
            "preferences": self.preferences,
            "summary": self.summary,
            "dialogues": [
                {
                    "timestamp": e.timestamp,
                    "speaker": e.speaker,
                    "text": e.text,
                    "language": e.language,
                    "emotion": e.emotion,
                }
                for e in self.dialogues
            ],
        }

    @staticmethod
    def from_dict(d: dict, track_id: int = 0) -> "Person":
        dialogues = [DialogueEntry(**e) for e in d.get("dialogues", [])]
        return Person(
            track_id=track_id,
            persistent_id=d.get("persistent_id", d.get("name", "")),
            name=d.get("name"),
            first_met=d.get("first_met", 0.0),
            last_seen=d.get("last_seen", 0.0),
            last_talked=d.get("last_talked", 0.0),
            times_seen=d.get("times_seen", 0),
            emotion=d.get("emotion", ""),
            dialogues=dialogues,
            facts=d.get("facts", []),
            preferences=d.get("preferences", {}),
            summary=d.get("summary", ""),
        )

    @property
    def first_met_dt(self) -> Optional[datetime]:
        return datetime.fromtimestamp(self.first_met) if self.first_met else None

    @property
    def last_seen_dt(self) -> Optional[datetime]:
        return datetime.fromtimestamp(self.last_seen) if self.last_seen else None

    @property
    def last_talked_dt(self) -> Optional[datetime]:
        return datetime.fromtimestamp(self.last_talked) if self.last_talked else None

    def time_since_seen(self) -> Optional[float]:
        if not self.last_seen:
            return None
        return time.time() - self.last_seen

    def time_since_talked(self) -> Optional[float]:
        if not self.last_talked:
            return None
        return time.time() - self.last_talked


class PeopleMemory:
    """In-memory people database with JSON file persistence.

    Primary lookup is by track_id (current session).
    Named people are persisted to disk and restored when re-identified.
    """

    def __init__(self, storage_dir: str = "people"):
        self._dir = storage_dir
        # Active session: track_id -> Person
        self._active: dict[int, Person] = {}
        # Persistent store: persistent_id -> Person dict (loaded from disk)
        self._stored: dict[str, dict] = {}
        self._lock = threading.Lock()

    def load(self):
        """Load all person files from disk into the persistent store."""
        os.makedirs(self._dir, exist_ok=True)
        count = 0
        for fname in os.listdir(self._dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(self._dir, fname)
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
                pid = data.get("persistent_id") or data.get("name") or fname[:-5]
                self._stored[pid.lower()] = data
                count += 1
            except Exception as e:
                logger.warning(f"Failed to load {fpath}: {e}")
        logger.info(f"People memory loaded: {count} people from {self._dir}/")

    # --- Lookup by track_id ---

    def get(self, track_id: int) -> Optional[Person]:
        """Get person by current session track_id."""
        with self._lock:
            return self._active.get(track_id)

    def get_or_create(self, track_id: int) -> Person:
        """Get or create a person for this track_id."""
        with self._lock:
            if track_id not in self._active:
                now = time.time()
                self._active[track_id] = Person(
                    track_id=track_id,
                    first_met=now,
                    last_seen=now,
                    times_seen=1,
                )
            return self._active[track_id]

    def identify(self, track_id: int, name: str):
        """Associate a track_id with a name. Loads persistent data if available."""
        person = self.get_or_create(track_id)
        pid = name.lower()

        # Load persistent history if this is a known person
        with self._lock:
            stored = self._stored.get(pid)

        if stored and not person.is_identified:
            # Merge stored data into the active person
            person.name = name
            person.persistent_id = pid
            person.facts = stored.get("facts", [])
            person.preferences = stored.get("preferences", {})
            person.summary = stored.get("summary", "")
            # Restore dialogue history from disk
            person.dialogues = [
                DialogueEntry(**e) for e in stored.get("dialogues", [])
            ]
            person.times_seen = stored.get("times_seen", 0) + 1
            stored_first = stored.get("first_met", 0.0)
            if stored_first and stored_first < person.first_met:
                person.first_met = stored_first
            logger.info(f"Identified track {track_id} as {name} (restored {len(person.dialogues)} dialogues, {len(person.facts)} facts)")
        else:
            person.name = name
            person.persistent_id = pid
            if not person.is_identified:
                logger.info(f"Identified track {track_id} as {name} (new person)")

        self._save(person)

    def remove_track(self, track_id: int):
        """Remove a track_id from active session (e.g. face disappeared).
        Persists data if person was identified."""
        with self._lock:
            person = self._active.pop(track_id, None)
        if person and person.is_identified:
            self._save(person)

    # --- Lookup by name (for persistent queries) ---

    def get_by_name(self, name: str) -> Optional[Person]:
        """Find an active person by name, or load from disk."""
        # Check active first
        with self._lock:
            for p in self._active.values():
                if p.name and p.name.lower() == name.lower():
                    return p
        # Fall back to stored
        stored = self._stored.get(name.lower())
        if stored:
            return Person.from_dict(stored)
        return None

    @property
    def active_ids(self) -> list[int]:
        with self._lock:
            return list(self._active.keys())

    @property
    def known_names(self) -> list[str]:
        """All names that have been persisted."""
        return [d.get("name", k) for k, d in self._stored.items() if d.get("name")]

    @property
    def active_count(self) -> int:
        return len(self._active)

    # --- Updates (all by track_id) ---

    def update_seen(self, track_id: int, emotion: str = ""):
        """Record that we saw this track."""
        person = self.get_or_create(track_id)
        now = time.time()
        if person.last_seen == 0 or (now - person.last_seen) > 300:
            person.times_seen += 1
        person.last_seen = now
        if emotion:
            person.emotion = emotion
            person.preferences["last_emotion"] = emotion

    def add_dialogue(self, track_id: int, speaker: str, text: str,
                     language: str = "", emotion: str = ""):
        """Add a dialogue entry."""
        person = self.get_or_create(track_id)
        person.dialogues.append(DialogueEntry(
            timestamp=time.time(), speaker=speaker, text=text,
            language=language, emotion=emotion,
        ))
        person.last_talked = time.time()
        if person.is_identified:
            self._save(person)

    def add_fact(self, track_id: int, fact: str):
        """Add a fact about a person (deduplicates)."""
        person = self.get_or_create(track_id)
        if fact not in person.facts:
            person.facts.append(fact)
            logger.info(f"New fact about track {track_id} ({person.name or '?'}): {fact}")
            if person.is_identified:
                self._save(person)

    def set_preference(self, track_id: int, key: str, value):
        person = self.get_or_create(track_id)
        person.preferences[key] = value

    def update_summary(self, track_id: int, summary: str):
        person = self.get_or_create(track_id)
        person.summary = summary
        if person.is_identified:
            self._save(person)

    # --- Context for LLM ---

    def get_context_for_llm(self, track_id: int, max_dialogues: int = 20) -> str:
        """Format a person's memory as context text for an LLM prompt."""
        person = self.get(track_id)
        if not person:
            return f"Unknown person (track {track_id}), no history."

        lines = []
        if person.name:
            lines.append(f"Person: {person.name}")
        else:
            lines.append(f"Unknown person (track {person.track_id})")

        if person.first_met:
            lines.append(f"First met: {person.first_met_dt.strftime('%Y-%m-%d %H:%M')}")

        ago = person.time_since_seen()
        if ago is not None:
            if ago < 60:
                lines.append(f"Last seen: {ago:.0f} seconds ago")
            elif ago < 3600:
                lines.append(f"Last seen: {ago/60:.0f} minutes ago")
            elif ago < 86400:
                lines.append(f"Last seen: {ago/3600:.1f} hours ago")
            else:
                lines.append(f"Last seen: {ago/86400:.1f} days ago")

        talk_ago = person.time_since_talked()
        if talk_ago is not None:
            if talk_ago < 3600:
                lines.append(f"Last talked: {talk_ago/60:.0f} minutes ago")
            else:
                lines.append(f"Last talked: {talk_ago/3600:.1f} hours ago")

        lines.append(f"Times seen: {person.times_seen}")

        if person.emotion:
            lines.append(f"Current emotion: {person.emotion}")

        if person.facts:
            lines.append(f"Known facts: {'; '.join(person.facts)}")

        if person.preferences:
            prefs = ", ".join(f"{k}={v}" for k, v in person.preferences.items())
            lines.append(f"Preferences: {prefs}")

        if person.summary:
            lines.append(f"Relationship summary: {person.summary}")

        recent = person.dialogues[-max_dialogues:]
        if recent:
            lines.append(f"Recent conversation ({len(recent)} exchanges):")
            for d in recent:
                ts = datetime.fromtimestamp(d.timestamp).strftime("%H:%M")
                who = "Them" if d.speaker == "person" else "Us"
                lang_tag = f" [{d.language}]" if d.language else ""
                emo_tag = f" ({d.emotion})" if d.emotion else ""
                lines.append(f"  [{ts}] {who}{lang_tag}{emo_tag}: {d.text}")

        return "\n".join(lines)

    def get_short_context(self, track_id: int) -> str:
        """One-line summary."""
        person = self.get(track_id)
        if not person:
            return f"Unknown track {track_id}"
        parts = []
        if person.name:
            parts.append(person.name)
        else:
            parts.append(f"track#{person.track_id}")
        if person.times_seen > 1:
            parts.append(f"seen {person.times_seen}x")
        if person.emotion:
            parts.append(person.emotion)
        if person.facts:
            parts.append(person.facts[0])
        return ", ".join(parts)

    # --- Persistence ---

    def _save(self, person: Person):
        """Save a person to disk (only if identified)."""
        if not person.is_identified or not person.persistent_id:
            return
        data = person.to_dict()
        pid = person.persistent_id.lower()
        # Update stored cache
        with self._lock:
            self._stored[pid] = data
        os.makedirs(self._dir, exist_ok=True)
        fpath = os.path.join(self._dir, f"{pid}.json")
        with open(fpath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_all(self):
        """Save all identified active people."""
        with self._lock:
            people = [p for p in self._active.values() if p.is_identified]
        for p in people:
            self._save(p)


# ---------------------------------------------------------------------------
# Standalone: inspect/manage people memory from CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="People memory CLI")
    parser.add_argument("--dir", default="people", help="Storage directory")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List all known people")
    show_p = sub.add_parser("show", help="Show a person's full memory")
    show_p.add_argument("name")
    context_p = sub.add_parser("context", help="Show LLM context for a person")
    context_p.add_argument("name")
    fact_p = sub.add_parser("add-fact", help="Add a fact about a person")
    fact_p.add_argument("name")
    fact_p.add_argument("fact")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    mem = PeopleMemory(storage_dir=args.dir)
    mem.load()

    if args.command == "list" or args.command is None:
        names = mem.known_names
        if not names:
            print("No people in memory.")
        else:
            print(f"{len(names)} people:\n")
            for name in sorted(names):
                person = mem.get_by_name(name)
                if person:
                    parts = [name]
                    if person.times_seen > 1:
                        parts.append(f"seen {person.times_seen}x")
                    if person.facts:
                        parts.append(person.facts[0])
                    ago = person.time_since_seen()
                    if ago and ago > 3600:
                        parts.append(f"last {ago/3600:.0f}h ago")
                    print(f"  {', '.join(parts)}")

    elif args.command == "show":
        person = mem.get_by_name(args.name)
        if not person:
            print(f"No person named '{args.name}'")
        else:
            print(json.dumps(person.to_dict(), indent=2, ensure_ascii=False))

    elif args.command == "context":
        person = mem.get_by_name(args.name)
        if not person:
            print(f"No person named '{args.name}'")
        else:
            # Use a fake track_id for display
            person.track_id = -1
            mem._active[-1] = person
            print(mem.get_context_for_llm(-1))

    elif args.command == "add-fact":
        person = mem.get_by_name(args.name)
        if not person:
            print(f"No person named '{args.name}', creating...")
            mem.get_or_create(0)
            mem.identify(0, args.name)
            person = mem.get(0)
        else:
            mem._active[0] = person
        mem.add_fact(0 if person.track_id == 0 else person.track_id, args.fact)
        print(f"Added fact about {args.name}: {args.fact}")


if __name__ == "__main__":
    main()
