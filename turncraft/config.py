"""Typed settings boundary. Offline execution does not require credentials."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# --- Environment variable names -------------------------------------------------------

ENV_API_KEY = "ANTHROPIC_API_KEY"
ENV_ASSISTANT_MODEL = "ASSISTANT_MODEL"
ENV_USER_MODEL = "USER_SIM_MODEL"
ENV_ASSISTANT_MAX_TOKENS = "ASSISTANT_MAX_TOKENS"
ENV_USER_MAX_TOKENS = "USER_SIM_MAX_TOKENS"
ENV_REQUEST_TIMEOUT = "LLM_REQUEST_TIMEOUT_SECONDS"
ENV_MAX_RETRIES = "LLM_MAX_RETRIES"
ENV_MAX_DIALOGUE_TURNS = "MAX_DIALOGUE_TURNS"
ENV_MAX_TOOL_ROUNDS = "MAX_TOOL_ROUNDS_PER_TURN"

# --- Model defaults -------------------------------------------------------------------

# Provider availability is account-dependent; live users must choose their model ids.
DEFAULT_ASSISTANT_MODEL = ""
DEFAULT_USER_MODEL = ""

# --- Token budgets --------------------------------------------------------------------
# max_tokens is REQUIRED on every Messages call, so these are floors, not optimisations.

# One short reply plus at most one <tool_call> envelope. Generous enough that a truncated
# tag means the model rambled, not that the budget was too tight.
DEFAULT_ASSISTANT_MAX_TOKENS = 2048
# One customer sentence plus the {message, done, reason} JSON envelope.
DEFAULT_USER_MAX_TOKENS = 512

# --- Loop and transport budgets -------------------------------------------------------

# Roughly twice the oracle dialogue length of the longest canonical task.
DEFAULT_MAX_DIALOGUE_TURNS = 8
# Covers read -> read -> write plus two recovery attempts after a policy denial.
DEFAULT_MAX_TOOL_ROUNDS_PER_TURN = 6
# Two identical failing actions with no intervening state change or user message is the
# deterministic no-progress signal; a third adds nothing but latency in a live demo.
NO_PROGRESS_REPEAT_LIMIT = 2
# Transient 429/5xx only. Bounded so a provider outage terminates the episode with a
# recorded reason instead of hanging the demo.
DEFAULT_MAX_RETRIES = 3
# Longer than any single short-completion call; short enough to fail visibly on stage.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0

# --- .env discovery -------------------------------------------------------------------

DOTENV_FILENAME = ".env"
# Comment marker recognised by the minimal .env reader below.
_DOTENV_COMMENT = "#"


class MissingApiKeyError(RuntimeError):
    """Raised when a live provider call is attempted without ``ANTHROPIC_API_KEY``."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable, process-wide configuration. Construct via :func:`get_settings`."""

    # repr=False: a frozen dataclass repr lands in tracebacks and log lines.
    anthropic_api_key: str = field(repr=False, default="")
    assistant_model: str = DEFAULT_ASSISTANT_MODEL
    user_model: str = DEFAULT_USER_MODEL
    assistant_max_tokens: int = DEFAULT_ASSISTANT_MAX_TOKENS
    user_max_tokens: int = DEFAULT_USER_MAX_TOKENS
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    max_dialogue_turns: int = DEFAULT_MAX_DIALOGUE_TURNS
    max_tool_rounds_per_turn: int = DEFAULT_MAX_TOOL_ROUNDS_PER_TURN
    no_progress_repeat_limit: int = NO_PROGRESS_REPEAT_LIMIT

    @property
    def has_api_key(self) -> bool:
        """True when a live run is possible. Callers branch on this, never on the key itself."""
        return bool(self.anthropic_api_key)


def _read_dotenv(path: Path) -> dict[str, str]:
    """Parse ``KEY=value`` lines from a .env file. Returns empty on any failure.

    Deliberately minimal: ``python-dotenv`` is not an allowed dependency. Values are
    never logged.
    """
    if not path.is_file():
        return {}
    parsed: dict[str, str] = {}
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        LOGGER.warning("Could not read %s (%s); continuing with process environment only.", path, exc)
        return {}
    except UnicodeDecodeError as exc:
        LOGGER.warning("%s is not valid UTF-8 (%s); continuing with process environment only.", path, exc)
        return {}
    for lineno, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(_DOTENV_COMMENT):
            continue
        key, separator, value = stripped.partition("=")
        if not separator:
            LOGGER.warning("Ignoring %s line %d: no '=' separator.", path, lineno)
            continue
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if not key:
            LOGGER.warning("Ignoring %s line %d: empty key.", path, lineno)
            continue
        parsed[key] = value.strip().strip("'\"")
    return parsed


def _candidate_dotenv_paths() -> list[Path]:
    """The environment root (parent of this package) first, then the current directory."""
    package_root = Path(__file__).resolve().parent.parent
    return [package_root / DOTENV_FILENAME, Path.cwd() / DOTENV_FILENAME]


def _collect_environment() -> dict[str, str]:
    """Process environment, with .env filling only the names it does not already define."""
    env = dict(os.environ)
    for path in _candidate_dotenv_paths():
        for key, value in _read_dotenv(path).items():
            env.setdefault(key, value)
    return env


def _env_str(env: dict[str, str], name: str, default: str) -> str:
    raw = env.get(name)
    return raw.strip() if raw and raw.strip() else default


def _env_int(env: dict[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if not raw or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        LOGGER.warning("%s=%r is not an integer; using default %d.", name, raw, default)
        return default


def _env_float(env: dict[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if not raw or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        LOGGER.warning("%s=%r is not a number; using default %s.", name, raw, default)
        return default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build the process-wide settings object. Cached, so the environment is read once."""
    env = _collect_environment()
    return Settings(
        anthropic_api_key=_env_str(env, ENV_API_KEY, ""),
        assistant_model=_env_str(env, ENV_ASSISTANT_MODEL, DEFAULT_ASSISTANT_MODEL),
        user_model=_env_str(env, ENV_USER_MODEL, DEFAULT_USER_MODEL),
        assistant_max_tokens=_env_int(env, ENV_ASSISTANT_MAX_TOKENS, DEFAULT_ASSISTANT_MAX_TOKENS),
        user_max_tokens=_env_int(env, ENV_USER_MAX_TOKENS, DEFAULT_USER_MAX_TOKENS),
        request_timeout_seconds=_env_float(env, ENV_REQUEST_TIMEOUT, DEFAULT_REQUEST_TIMEOUT_SECONDS),
        max_retries=_env_int(env, ENV_MAX_RETRIES, DEFAULT_MAX_RETRIES),
        max_dialogue_turns=_env_int(env, ENV_MAX_DIALOGUE_TURNS, DEFAULT_MAX_DIALOGUE_TURNS),
        max_tool_rounds_per_turn=_env_int(env, ENV_MAX_TOOL_ROUNDS, DEFAULT_MAX_TOOL_ROUNDS_PER_TURN),
        no_progress_repeat_limit=NO_PROGRESS_REPEAT_LIMIT,
    )


def require_api_key(settings: Settings | None = None) -> str:
    """Return the API key, or raise with an actionable message. Never logs the value."""
    resolved = settings if settings is not None else get_settings()
    if not resolved.anthropic_api_key:
        raise MissingApiKeyError(
            f"{ENV_API_KEY} is not set. Export it, or place it in a {DOTENV_FILENAME} file "
            f"next to requirements.txt. Offline tests and scripted demos do not need it."
        )
    return resolved.anthropic_api_key


__all__ = [
    "DEFAULT_ASSISTANT_MAX_TOKENS",
    "DEFAULT_ASSISTANT_MODEL",
    "DEFAULT_MAX_DIALOGUE_TURNS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_TOOL_ROUNDS_PER_TURN",
    "DEFAULT_REQUEST_TIMEOUT_SECONDS",
    "DEFAULT_USER_MAX_TOKENS",
    "DEFAULT_USER_MODEL",
    "ENV_API_KEY",
    "ENV_ASSISTANT_MAX_TOKENS",
    "ENV_ASSISTANT_MODEL",
    "ENV_MAX_DIALOGUE_TURNS",
    "ENV_MAX_RETRIES",
    "ENV_MAX_TOOL_ROUNDS",
    "ENV_REQUEST_TIMEOUT",
    "ENV_USER_MAX_TOKENS",
    "ENV_USER_MODEL",
    "MissingApiKeyError",
    "NO_PROGRESS_REPEAT_LIMIT",
    "Settings",
    "get_settings",
    "require_api_key",
]
