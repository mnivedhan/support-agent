"""LLM access through the Claude Code CLI (`claude -p`), billed to your Claude Code login instead of an API key.

Every stage calls `complete()` and nothing else, so the backend can later be swapped for the Anthropic API
(needed for streaming and high concurrency in later stages) without touching stage code.

With `schema`, the CLI forces the answer to match a JSON Schema and returns it parsed in `data`. Under the hood
it does this with a hidden tool call, which is why such calls report two turns.
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResult:
    text: str
    stop_reason: str
    model: str
    session_id: str
    input_tokens: int  # uncached + cache writes + cache reads: everything the model read
    cache_read_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float  # list-price equivalent reported by the CLI; not billed on a subscription
    data: Any = None  # the parsed structured output, when called with a schema


class LLMError(RuntimeError):
    pass


def complete(system: str, user: str, model: str = "claude-opus-5-5", effort: str = "medium",
             timeout_s: float = 300, schema: dict | None = None) -> LLMResult:
    cmd = [
        "claude", "-p",
        "--output-format", "json",
        "--model", model,
        "--effort", effort,
        "--system-prompt", system,   # replaces Claude Code's own coding-agent prompt
        "--tools", "",               # a plain completion: no file, shell or web access
        "--strict-mcp-config",       # don't load your MCP servers
        "--no-session-persistence",  # don't fill your session history with test runs
    ]
    if schema is not None:
        cmd += ["--json-schema", json.dumps(schema)]
    # With ANTHROPIC_API_KEY set the CLI would bill that key; drop it so your Claude Code login is used.
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    # Run outside the repo so no CLAUDE.md or git status leaks into the prompt.
    with tempfile.TemporaryDirectory() as cwd:
        proc = subprocess.run(cmd, input=user, capture_output=True, text=True, env=env, cwd=cwd,
                              timeout=timeout_s, check=False)  # errors arrive as JSON, parsed below

    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise LLMError(f"claude exited {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}") from e
    if out.get("is_error"):
        raise LLMError(f"claude returned an error ({out.get('subtype')}): {out.get('result')}")

    usage = out["usage"]
    return LLMResult(
        text=out["result"],
        stop_reason=out.get("stop_reason", ""),
        model=next(iter(out.get("modelUsage", {})), model),
        session_id=out["session_id"],
        input_tokens=usage["input_tokens"] + usage["cache_creation_input_tokens"] + usage["cache_read_input_tokens"],
        cache_read_tokens=usage["cache_read_input_tokens"],
        output_tokens=usage["output_tokens"],
        latency_s=round(out["duration_ms"] / 1000, 2),
        cost_usd=round(out["total_cost_usd"], 4),
        data=out.get("structured_output"),
    )
