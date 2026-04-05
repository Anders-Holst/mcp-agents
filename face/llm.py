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
import asyncio
import threading
import logging
from datetime import datetime
from typing import Optional

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from people_memory import PeopleMemory

logger = logging.getLogger("llm")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are {name}, a camera-based assistant that can see, hear, and speak.

What you perceive:
- You see people through a camera (face recognition, emotion detection)
- You hear speech through a microphone
- You remember people you've met (names, past conversations, facts)
{capabilities}
Current time: {time}

Rules: Reply in 1-2 short sentences. Match their language. \
No markdown or emojis. Keep spoken responses natural and conversational."""

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
                 agent_name: str = "Face Agent"):
        provider = OpenAIProvider(base_url=ollama_url, api_key="ollama")
        model = OpenAIChatModel(model_name, provider=provider)
        self._model = model
        self._model_name = model_name
        self._mcp_servers = mcp_servers or []
        self._agent_name = agent_name

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
            model_settings=ModelSettings(extra_body={"think": False}),
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
                          emotion: str = "") -> str:
        """Generate a contextual greeting using the LLM."""
        context = memory.get_context_for_llm(track_id, max_dialogues=5)
        person = memory.get(track_id)
        name = person.name if person else "someone"

        prompt = f"""Greet this person. One sentence.
{context}
Emotion: {emotion or 'neutral'}"""

        fallback = f"Hello, {name}!" if name != "someone" else "Hello there!"
        return self._call_llm(prompt, "greeting", fallback)

    def generate_response(self, memory: PeopleMemory, track_id: Optional[int],
                          heard_text: str, language: str = "en") -> str:
        """Generate a response to speech using the LLM."""
        if track_id:
            context = memory.get_context_for_llm(track_id, max_dialogues=5)
        else:
            context = "Unknown person."

        prompt = f"""They said: "{heard_text}" (language: {language})
{context}
Reply in their language. One sentence."""

        return self._call_llm(prompt, "response", f"I heard you say: {heard_text}")

    def generate_ask_name(self, track_id: int) -> str:
        """Generate a natural way to ask someone's name."""
        prompt = "An unknown person just appeared on camera. Ask them for their name in a friendly way. One short sentence."
        return self._call_llm(prompt, "ask_name",
                              "Hello! I don't think we've met. What is your name?")
