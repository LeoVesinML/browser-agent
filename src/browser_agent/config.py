"""Single place where every knob lives, so nothing is hard-coded deeper down."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# override=True on purpose: a project-local .env is an explicit choice, and the
# most common failure otherwise is a stale ANTHROPIC_BASE_URL exported in the
# user's shell silently winning over the endpoint they just configured here.
load_dotenv(override=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelConfig:
    # The model that plans and acts.
    main: str = os.getenv("AGENT_MODEL", "claude-opus-5")
    # A cheaper model for bounded, mechanical sub-calls (risk judging, compaction).
    small: str = os.getenv("AGENT_SMALL_MODEL", "claude-haiku-4-5")
    effort: str = os.getenv("AGENT_EFFORT", "high")  # low | medium | high | xhigh | max
    max_tokens: int = 8_000
    # "anthropic" uses the full Messages API surface (adaptive thinking, effort).
    # "compatible" targets an Anthropic-compatible third-party endpoint such as
    # z.ai's GLM, which accepts the message/tool shape but not the newer fields.
    profile: str = os.getenv("AGENT_API_PROFILE", "anthropic")
    base_url: str | None = os.getenv("ANTHROPIC_BASE_URL") or None


@dataclass
class ContextConfig:
    """Everything that keeps the conversation from growing without bound."""

    # Hard ceiling we manage against - deliberately far below the model's window,
    # because latency and cost scale with what we actually send.
    budget_tokens: int = int(os.getenv("AGENT_CONTEXT_BUDGET", "70000"))
    # Fraction of the budget at which we compact the transcript.
    compact_at: float = 0.75
    # How many page observations stay verbatim; older ones collapse to one line.
    keep_observations: int = 3
    # How many assistant/tool turns survive compaction untouched.
    keep_recent_turns: int = 6
    # Per-observation caps.
    snapshot_chars: int = 6_000
    read_chars: int = 4_000
    tool_result_chars: int = 8_000
    verify_tokens_every: int = 6


@dataclass
class SafetyConfig:
    # ask     - confirm anything risky (default)
    # yolo    - never ask (for unattended demos on throwaway data)
    # strict  - confirm anything that mutates state at all
    mode: str = os.getenv("AGENT_SAFETY", "ask")
    use_llm_judge: bool = os.getenv("AGENT_LLM_JUDGE", "1") != "0"


@dataclass
class BrowserConfig:
    profile_dir: Path = field(
        default_factory=lambda: Path(os.getenv("AGENT_PROFILE_DIR", PROJECT_ROOT / "profiles" / "default"))
    )
    headless: bool = os.getenv("AGENT_HEADLESS", "0") == "1"
    start_url: str = os.getenv("AGENT_START_URL", "about:blank")
    slow_mo_ms: int = int(os.getenv("AGENT_SLOW_MO", "0"))
    record_video: bool = False


@dataclass
class AgentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    browser: BrowserConfig = field(default_factory=BrowserConfig)
    max_steps: int = int(os.getenv("AGENT_MAX_STEPS", "60"))
    subagent_max_steps: int = 14
    runs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "runs")
