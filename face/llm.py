"""
LLM conversation engine using pydantic-ai + Ollama.

Handles system prompt construction, MCP toolset wiring, and async
execution.  The Agent class calls the high-level methods here
(generate_greeting, generate_response, generate_ask_name) without
needing to know about pydantic-ai internals.

To change the agent's personality or add new generation methods,
edit this file.
"""

import time
import random
import asyncio
import threading
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from pydantic_ai import Agent as PydanticAgent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from people_memory import PeopleMemory

logger = logging.getLogger("llm")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Canned greeting templates (used when smart_greetings=False)
# ---------------------------------------------------------------------------

_PLAIN_GREETINGS = [
    "Hello, {name}!",
    "Hi {name}!",
    "Hey {name}, good to see you!",
    "Welcome back, {name}!",
    "Hi again, {name}!",
]

_INTERVIEW_GREETINGS: dict[str, list[str]] = {
    "favourite_colour": [
        "Hi {name}! What's your favourite colour?",
        "Hey {name}, what colour do you like best?",
        "Hello {name}! Got a favourite colour?",
    ],
    "hobby": [
        "Hi {name}! What do you like doing for fun?",
        "Hey {name}, got any hobbies you enjoy?",
        "Hello {name}! What do you do in your spare time?",
    ],
    "favourite_food": [
        "Hi {name}! What's your favourite food?",
        "Hey {name}, what do you love to eat?",
        "Hello {name}! Got a favourite dish?",
    ],
    "favourite_music": [
        "Hi {name}! What music are you into?",
        "Hey {name}, any favourite artists these days?",
        "Hello {name}! What do you like listening to?",
    ],
}

_ASK_NAME_PROMPTS = [
    "Hello! What's your name?",
    "Hi there, I don't think we've met — what should I call you?",
    "Hey! What's your name?",
    "Hi! I don't recognise you — what's your name?",
]


SYSTEM_PROMPT = """\
You are {name}, a camera-based assistant that can see, hear, and speak.

What you perceive:
- You see people through a camera (face recognition, emotion detection)
- You hear speech through a microphone
- You remember people you've met (names, past conversations, facts about them)
{capabilities}
Current time: {time}

Rules:
- Reply in 1-2 short sentences. Match their language.
- Address the person by name when you know it (but don't overdo it).
- When it fits naturally, reference what you already know about them
  (their job, hobbies, likes, past conversations). Don't force it —
  only when it genuinely makes the reply more personal.
- No markdown or emojis. Keep spoken responses natural and conversational."""

# ---------------------------------------------------------------------------
# Tool deps — passed to pydantic-ai tools via RunContext
# ---------------------------------------------------------------------------

@dataclass
class ConversationDeps:
    """Dependencies injected into pydantic-ai tool calls."""
    memory: PeopleMemory
    track_id: int
    person_id: str = ""


# ---------------------------------------------------------------------------
# ConversationLLM
# ---------------------------------------------------------------------------

class ConversationLLM:
    """LLM-powered conversation engine.

    Supports MCP toolsets so the LLM can call external tools (smart home,
    search, etc.).  Pass MCP servers via the ``mcp_servers`` parameter —
    see ``mcp_client.py`` for how to configure them.
    """

    def __init__(self, model_name: str = "qwen3:8b",
                 ollama_url: str = "http://localhost:11434/v1",
                 mcp_servers: Optional[list] = None,
                 mcp_descriptions: Optional[list[str]] = None,
                 agent_name: str = "Face Agent",
                 smart_greetings: bool = False):
        provider = OpenAIProvider(base_url=ollama_url, api_key="ollama")
        model = OpenAIChatModel(model_name, provider=provider)
        self._model = model
        self._model_name = model_name
        self._mcp_servers = mcp_servers or []
        self._agent_name = agent_name
        self._smart_greetings = smart_greetings

        # Build capabilities text from MCP descriptions
        if mcp_descriptions:
            lines = "\n".join(f"- {d}" for d in mcp_descriptions)
            self._capabilities = f"\nYour tools:\n{lines}\n"
        else:
            self._capabilities = ""

        # Background event loop for async MCP operations
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="llm-loop")
        self._loop_thread.start()

        if self._mcp_servers:
            logger.info(f"ConversationLLM: {len(self._mcp_servers)} MCP toolset(s) active")

    def stop(self):
        """Shut down the background event loop."""
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5)

    def _make_agent(self) -> PydanticAgent:
        """Create a pydantic-ai agent with current system prompt + MCP toolsets."""
        system = SYSTEM_PROMPT.format(
            name=self._agent_name,
            time=datetime.now().strftime("%H:%M"),
            capabilities=self._capabilities,
        )
        return PydanticAgent(
            self._model,
            system_prompt=system,
            toolsets=self._mcp_servers,
            model_settings=ModelSettings(extra_body={
                "reasoning_effort": "none",
                "think": False,
            }),
        )

    # --- Core LLM call (sync wrapper around async) ---

    def _call_llm(self, prompt: str, label: str, fallback: str) -> str:
        """Call the LLM (with MCP tools if configured). Thread-safe."""
        logger.info(f"[LLM:{label}] model={self._model_name}")
        logger.info(f"[LLM:{label}] prompt:\n{prompt}")
        start = time.time()
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._arun(prompt), self._loop)
            output = future.result(timeout=60)
            elapsed = time.time() - start
            logger.info(f"[LLM:{label}] response ({elapsed:.1f}s): {output}")
            return output
        except Exception as e:
            elapsed = time.time() - start
            logger.warning(f"[LLM:{label}] FAILED after {elapsed:.1f}s: {e}")
            logger.info(f"[LLM:{label}] using fallback: {fallback}")
            return fallback

    async def _arun(self, prompt: str) -> str:
        """Run a single LLM query, connecting MCP servers if needed."""
        agent = self._make_agent()
        if self._mcp_servers:
            async with agent:
                result = await agent.run(prompt)
                return result.output.strip()
        else:
            result = await agent.run(prompt)
            return result.output.strip()

    # --- High-level generation methods ---

    def generate_greeting(self, memory: PeopleMemory, track_id: int,
                          emotion: str = "",
                          interview_topic: Optional[str] = None) -> str:
        """Return a greeting for ``track_id``.

        By default picks a random canned template — instant, no LLM call.
        If ``smart_greetings=True`` was set at init time, goes through the
        LLM so facts can be woven in.
        """
        person = memory.get(track_id)
        name = person.name if person else "someone"

        if not self._smart_greetings:
            return self._canned_greeting(name, interview_topic)

        context = memory.get_context_for_llm(track_id, max_dialogues=5)
        if interview_topic:
            prompt = f"""Greet this person warmly by name, then in the same reply ask one friendly, natural question about their {interview_topic.replace('_', ' ')}.
Keep the whole reply to 1-2 short sentences. No markdown or emojis.
{context}
Emotion: {emotion or 'neutral'}"""
        else:
            prompt = f"""Greet this person by name in one short sentence. If you know something interesting about them from the context below (facts, recent conversation), you may weave it in — but only when it fits naturally.
{context}
Emotion: {emotion or 'neutral'}"""

        fallback = self._canned_greeting(name, interview_topic)
        return self._call_llm(prompt, "greeting", fallback)

    def _canned_greeting(self, name: str,
                         interview_topic: Optional[str]) -> str:
        """Pick a random template for the given situation."""
        if name == "someone":
            return "Hello there!"
        if interview_topic:
            options = _INTERVIEW_GREETINGS.get(interview_topic)
            if options:
                return random.choice(options).format(name=name)
        return random.choice(_PLAIN_GREETINGS).format(name=name)

    def generate_response(self, memory: PeopleMemory, track_id: Optional[int],
                          heard_text: str, language: str = "en") -> str:
        """Generate a conversational response.

        Fact extraction happens separately in a background call (see
        ``extract_facts``). This method focuses on fast response generation.
        MCP toolsets are wired in if configured.
        """
        if track_id:
            context = memory.get_context_for_llm(track_id, max_dialogues=5)
        else:
            context = "Unknown person."

        prompt = f"""They said: "{heard_text}" (language: {language})
{context}

Reply naturally in their language in 1-2 short sentences. Address them by name when it fits. You may reference what you already know about them (facts, earlier conversation) if it makes the reply more personal — but only when it genuinely fits."""

        return self._call_llm(prompt, "response", f"I heard you say: {heard_text}")

    def extract_facts_with_tools(self, memory: PeopleMemory, track_id: int,
                                person_said: str, agent_said: str = ""):
        """Background: use tool calling to extract and store facts.

        Runs WITHOUT reasoning_effort:none so the model can call tools.
        Slower (~5-13s) but uses proper function calling. Meant to run
        in a background thread so the user doesn't wait.
        """
        person = memory.get(track_id)
        if not person:
            return
        person_id = person.persistent_id or ""
        context = memory.get_context_for_llm(track_id, max_dialogues=5)

        prompt = f"""Use the tools to store any personal facts from this conversation!

{context}

The person just said: "{person_said}"

Call write_fact for each NEW fact (job, hobbies, family, location, favourites,
likes/dislikes). Make facts self-contained ("Favourite food is sushi" not "sushi").
Call replace_fact if a fact UPDATES an old one (e.g. favourite food changed).
Call set_name if they stated or corrected their name.
If nothing new, say "Nothing new." """

        deps = ConversationDeps(
            memory=memory, track_id=track_id, person_id=person_id)
        logger.info(f"[LLM:tools] extracting facts with tool calling")
        start = time.time()
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._arun_with_tools(prompt, deps), self._loop)
            output = future.result(timeout=30)
            elapsed = time.time() - start
            logger.info(f"[LLM:tools] done ({elapsed:.1f}s): {output}")
        except Exception as e:
            elapsed = time.time() - start
            logger.warning(f"[LLM:tools] failed after {elapsed:.1f}s: {e}")

    def _make_tool_agent(self) -> PydanticAgent:
        """Create a pydantic-ai agent with write_fact, replace_fact, set_name.

        No reasoning_effort:none here — gemma4 needs thinking enabled to call tools.
        """
        agent = PydanticAgent(
            self._model,
            system_prompt="You extract personal facts from conversations. Use the tools to store facts about the user!",
            deps_type=ConversationDeps,
            toolsets=self._mcp_servers,
        )

        @agent.tool
        def write_fact(ctx: RunContext[ConversationDeps], fact: str) -> str:
            """Remember a personal fact about the person."""
            logger.info(f"[TOOL:write_fact] {fact}")
            ctx.deps.memory.add_fact(ctx.deps.track_id, fact)
            return "Stored."

        @agent.tool
        def replace_fact(ctx: RunContext[ConversationDeps], old_fact: str, new_fact: str) -> str:
            """Replace an outdated fact with an updated version."""
            logger.info(f"[TOOL:replace_fact] {old_fact!r} -> {new_fact!r}")
            ctx.deps.memory.replace_fact(ctx.deps.track_id, old_fact, new_fact)
            return "Replaced."

        @agent.tool
        def set_name(ctx: RunContext[ConversationDeps], name: str) -> str:
            """Update the person's name if they corrected or stated it."""
            logger.info(f"[TOOL:set_name] {name}")
            if ctx.deps.person_id:
                ctx.deps.memory.rename(ctx.deps.person_id, name)
            return "Name updated."

        return agent

    async def _arun_with_tools(self, prompt: str, deps: ConversationDeps) -> str:
        """Run the tool-calling agent (write_fact, replace_fact, set_name)."""
        agent = self._make_tool_agent()
        if self._mcp_servers:
            async with agent:
                result = await agent.run(prompt, deps=deps)
                return result.output.strip()
        else:
            result = await agent.run(prompt, deps=deps)
            return result.output.strip()

    def generate_ask_name(self, track_id: int) -> str:
        """Return a random 'what's your name?' prompt. Always canned."""
        return random.choice(_ASK_NAME_PROMPTS)

    def extract_name(self, person_said: str) -> Optional[str]:
        """Extract a personal name from speech via the LLM.

        The person was just asked 'what is your name?' and this is what
        they said.  Returns the properly capitalized name, or None.
        """
        if not person_said:
            return None
        prompt = f"""Someone was asked "What is your name?" and replied:

"{person_said}"

Extract their personal name from this reply.
Reply with ONLY the name, properly capitalized (e.g. "Joakim", "Anna-Karin").
If no name is present or you are unsure, reply NONE."""

        result = self._call_llm(prompt, "extract_name", "NONE")
        if not result:
            return None
        name = result.strip().strip('."\'').strip()
        if not name or name.upper() == "NONE" or len(name) > 60:
            return None
        return name

