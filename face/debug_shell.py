"""Interactive debug shell for poking at people memory (and, when
attached to a running Agent, at the live tracker / voice loop).

Used in two modes:
  - Standalone, via ``pixi run shell`` — reads people/ from disk.
  - Embedded, via ``pixi run agent --shell`` — runs as a daemon
    thread alongside the live agent and shares its PeopleMemory and
    FaceTracker instances, so edits take effect immediately.
"""

import json
import logging
from typing import Optional

from people_memory import PeopleMemory, INTERVIEW_TOPICS


MEMORY_HELP = """\
Memory commands:
  list                       List all known people
  show <name>                Dump a person's full JSON
  facts <name>               Show facts and asked-topics for a person
  context <name>             Show the LLM-facing context block
  missing <name>             List interview topics not yet asked
  reset-topics <name>        Clear asked_topics so questions get re-asked
  add-fact <name> <fact>     Append a free-form fact
  rename <old> <new>         Rename a person (display name only; ID is stable)
  delete <name>              Remove a person from memory and disk
"""

AGENT_HELP = """\
Agent commands (only when running inside the agent):
  tracks                     List active face tracks
  focus                      Show the currently-focused face
  greet <track_id>           Force a greeting (triggers interview question)
  ask <track_id>             Force asking for a name
  speak <text>               Speak text via TTS
  status                     Show agent busy/listener state (for debugging)
  pause / resume             Pause/resume speech listening
"""

GENERIC_HELP = """\
  help                       Show this message
  quit / exit                Leave the shell (does not stop the agent)
"""


def run_shell(mem: PeopleMemory, agent: Optional[object] = None):
    """Interactive REPL for poking at people memory.

    If ``agent`` is provided, additional commands become available for
    controlling the live agent (trigger greetings, inspect tracks, etc.).
    """
    # Mute the people_memory INFO lines so the prompt stays readable.
    logging.getLogger("people_memory").setLevel(logging.WARNING)

    help_text = MEMORY_HELP + (AGENT_HELP if agent else "") + GENERIC_HELP

    mode = "live agent" if agent else "standalone"
    print(f"People memory shell [{mode}]. "
          f"{len(mem.known_names)} people loaded from {mem._dir}/")
    print("Type 'help' for commands, 'quit' to exit.")

    SHELL_TID = -999  # fake track for name-addressed operations

    def _name_for_track(track_id) -> Optional[str]:
        person = mem.get(track_id)
        return person.name if person and person.is_identified else None

    def focused_name() -> Optional[str]:
        """Return the name of the currently-focused face, if any."""
        if not agent:
            return None
        face = agent.tracker.get_primary_face()
        if not face:
            return None
        return _name_for_track(face.track_id)

    def resolve_name(parts: list, idx: int = 1) -> Optional[str]:
        """Return the name at parts[idx], or the focused face's name if omitted."""
        if len(parts) > idx:
            return parts[idx]
        name = focused_name()
        if name:
            print(f"(using focused face: {name})")
            return name
        print("No name given and no focused face.")
        return None

    def find(name: str):
        person = mem.get_by_name(name)
        if not person:
            print(f"No person named '{name}'")
            return None
        mem._active[SHELL_TID] = person
        person.track_id = SHELL_TID
        return person

    while True:
        try:
            line = input("people> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue

        parts = line.split(maxsplit=3)
        cmd = parts[0].lower()

        # --- exit ---
        if cmd in ("quit", "exit", "q"):
            return
        elif cmd == "help":
            print(help_text)

        # --- memory commands ---
        elif cmd == "list":
            names = sorted(mem.known_names)
            if not names:
                print("(no people)")
            for name in names:
                person = mem.get_by_name(name)
                bits = [name]
                if person and person.times_seen > 1:
                    bits.append(f"seen {person.times_seen}x")
                missing = person.missing_topics() if person else []
                if missing:
                    bits.append(f"{len(missing)} topic(s) not asked")
                print("  " + ", ".join(bits))
        elif cmd == "show":
            name = resolve_name(parts)
            if name:
                person = mem.get_by_name(name)
                if not person:
                    print(f"No person named '{name}'")
                else:
                    print(json.dumps(person.to_dict(), indent=2, ensure_ascii=False))
        elif cmd == "facts":
            name = resolve_name(parts)
            if name:
                person = mem.get_by_name(name)
                if not person:
                    print(f"No person named '{name}'")
                else:
                    print(f"{person.name} (seen {person.times_seen}x)")
                    print("  Facts:")
                    if person.facts:
                        for f in person.facts:
                            print(f"    - {f}")
                    else:
                        print("    (none)")
                    print("  Asked topics:")
                    if person.asked_topics:
                        for t in person.asked_topics:
                            print(f"    - {t}")
                    else:
                        print("    (none)")
                    missing = person.missing_topics()
                    if missing:
                        print(f"  Not yet asked: {', '.join(missing)}")
        elif cmd == "context":
            name = resolve_name(parts)
            if name and find(name):
                print(mem.get_context_for_llm(SHELL_TID))
        elif cmd == "missing":
            name = resolve_name(parts)
            if name:
                person = mem.get_by_name(name)
                if not person:
                    print(f"No person named '{name}'")
                else:
                    missing = person.missing_topics()
                    if not missing:
                        print("All interview topics have been asked.")
                    else:
                        for t in missing:
                            print(f"  {t}")
        elif cmd == "reset-topics":
            name = resolve_name(parts)
            if name:
                person = find(name)
                if person:
                    n = len(person.asked_topics)
                    person.asked_topics = []
                    mem._save(person)
                    print(f"Cleared {n} asked topic(s) for {name}")
        elif cmd == "add-fact" and len(parts) >= 3:
            if find(parts[1]):
                mem.add_fact(SHELL_TID, parts[2])
                print(f"Added fact: {parts[2]}")
        elif cmd == "rename" and len(parts) >= 3:
            target = mem.get_by_name(parts[1])
            if target and mem.rename(target.persistent_id, parts[2]):
                print(f"Renamed {parts[1]} -> {parts[2]} ({target.persistent_id})")
            else:
                print(f"Could not rename {parts[1]}")
        elif cmd == "delete" and len(parts) >= 2:
            try:
                confirm = input(f"Really delete '{parts[1]}'? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            if confirm == "y":
                target = mem.get_by_name(parts[1])
                if target and mem.delete(target.persistent_id):
                    print(f"Deleted {parts[1]} ({target.persistent_id})")
                    if agent:
                        agent.tracker.db.remove_person(target.persistent_id)
                else:
                    print(f"No person named '{parts[1]}'")
            else:
                print("Aborted.")

        # --- agent commands ---
        elif agent and cmd == "tracks":
            faces = agent.tracker.get_visible_faces()
            if not faces:
                print("(no visible tracks)")
            for f in faces:
                name = _name_for_track(f.track_id) or "?"
                pid = agent.tracker.get_person_id(f.track_id) or "-"
                print(f"  track={f.track_id} id={pid} name={name} "
                      f"emotion={f.emotion} visible_frames={f.frames_visible}")
        elif agent and cmd == "greet" and len(parts) >= 2:
            try:
                tid = int(parts[1])
            except ValueError:
                print("greet needs an integer track_id")
                continue
            agent.greet(tid)
            print(f"Triggered greet for track {tid}")
        elif agent and cmd == "ask" and len(parts) >= 2:
            try:
                tid = int(parts[1])
            except ValueError:
                print("ask needs an integer track_id")
                continue
            agent.ask_name(tid)
            print(f"Triggered ask_name for track {tid}")
        elif agent and cmd == "speak" and len(parts) >= 2:
            text = line.split(maxsplit=1)[1]
            agent.speak(text)
        elif agent and cmd == "status":
            print("--- Agent ---")
            busy = agent._busy
            reason = agent._busy_reason
            since = time.time() - agent._busy_since if agent._busy_since else 0
            print(f"  busy:        {busy}" + (f" ({reason}, {since:.0f}s)" if busy else ""))
            print(f"  auto_ask:    {agent.auto_ask}")
            print(f"  auto_greet:  {agent.auto_greet}")

            print("--- Listener ---")
            listener = getattr(agent, "_listener", None)
            if listener:
                print(f"  paused:      {listener.paused}")
                print(f"  running:     {getattr(listener, '_running', '?')}")
            else:
                print(f"  (no listener)")
            vi = agent.voice_in
            print(f"  listen_phase:{vi.listen_phase or '(idle)'}")
            print(f"  vad_prob:    {vi.vad_prob:.3f}")
            print(f"  cancel_flag: {vi._cancel_listen}")

            print("--- TTS ---")
            vo = agent.voice_out
            print(f"  speaking:    {vo.speaking}")
            print(f"  ready:       {vo.ready}")
            print(f"  interrupted: {vo.interrupted}")
            tts_locked = not vo._lock.acquire(timeout=0)
            if not tts_locked:
                vo._lock.release()
            print(f"  lock_held:   {tts_locked}")

            print("--- AEC ---")
            echo = agent._echo_detector
            if echo:
                print(f"  raw_rms:     {echo.current_rms:.6f}")
                print(f"  clean_rms:   {echo.clean_rms:.6f}")
                print(f"  output_rms:  {echo.output_rms:.6f}")
                print(f"  threshold:   {echo._speech_threshold:.4f}")
                print(f"  user_speak:  {echo.user_speaking}")
                print(f"  playing:     {echo.playing}")
            else:
                print(f"  (not active)")

            print("--- Faces ---")
            faces = agent.tracker.get_visible_faces()
            print(f"  visible:     {len(faces)}")
            focus = agent.tracker.get_primary_face()
            if focus:
                print(f"  focus:       track={focus.track_id} "
                      f"name={_name_for_track(focus.track_id)!r} "
                      f"emotion={focus.emotion}")
            else:
                print(f"  focus:       (none)")

            print("--- Greeted ---")
            greeted = getattr(agent, "_greeted", {})
            if greeted:
                import time as _t
                for pid, ts in greeted.items():
                    ago = _t.time() - ts
                    print(f"  {pid}: {ago:.0f}s ago")
            else:
                print(f"  (none)")
        elif agent and cmd == "busy":
            print(f"busy={agent.busy}")
        elif agent and cmd == "focus":
            face = agent.tracker.get_primary_face()
            if not face:
                print("(no focused face)")
            else:
                print(f"  track={face.track_id} "
                      f"name={_name_for_track(face.track_id)!r} "
                      f"emotion={face.emotion}")
        elif agent and cmd == "pause":
            agent.pause_listening()
            print("Listening paused.")
        elif agent and cmd == "resume":
            agent.resume_listening()
            print("Listening resumed.")

        else:
            print(f"Unknown or incomplete command: {line!r}. Type 'help'.")
