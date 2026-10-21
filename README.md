# Store Support Agent, built stage by stage

Brightcart is a small online store. Customers write in: an order arrived broken, came late, never showed up, or
they want to return something. A support agent has to work out what kind of case it is, look up the policy,
calculate the refund or credit, decide whether a manager must approve it, act, and reply to the customer. Some
messages are just questions; some are attempts to trick the agent.

This repo builds that support agent from a single LLM call up to a fault-tolerant multi-agent system with a
production RAG pipeline. Every stage follows the same loop: **build the simplest version → measure it → break it
on purpose → fix the bottleneck you actually saw.**

It's a sister project of `~/Workspace/supply-chain-agents`, which uses the same stages on a harder domain and
has a finished reference design in its `reference/` folder.

## Roadmap

| # | Stage | Bottleneck it exposes, and the fix |
|---|---|---|
| 0 | **Baseline:** every policy + the store DB in one prompt, one call | Prompt size and cost grow with the data; made-up numbers; unstructured output |
| 1 | Simple RAG: fixed-size chunks, embeddings, cosine search | Eval doesn't look like real traffic; DB facts aren't in the message; rules split in half; look-alike docs |
| 1b | RAG fixes: heading chunks, hybrid BM25 + RRF, case-context queries, scope filters, pinned SOP | Measured one fix at a time; leaves vocabulary gaps and doc-level labels that hide section misses |
| 2 | Tool-using agent loop + structured output; code does the math | Model classifies, code calculates; schema + semantic + grounding checks with repair; graded automatically |
| 3 | Guardrails in code, case-scoped tools, budgets, a critic | 6 attacks: intake quarantine (0 LLM calls), sanitised tool data; the critic itself gets socially engineered |
| 4 | Manager approval + saved checkpoints | Approval lost on restart; payments repeated on recovery; approvals of stale data → SQLite checkpoints, `interrupt()`, idempotent ledger, re-check before executing |
| 5 | FastAPI service | LLM latency blocks requests → return 202, hand work to background workers |
| 6 | Redis Streams queue + multi-agent fleet (triage, resolver, comms) | Duplicate messages, crashed workers → idempotent refunds, retries, dead-letter queue |
| 7 | Production RAG at scale | 6 short policies hide every RAG problem → large help centre, pgvector/Qdrant, incremental ingest, caching |
| 8 | Observability + evals | Cost, latency and quality drift → tracing, metrics, CI checks |
| 9 | Scale and failover | Provider outages, rate limits → fallback models, circuit breakers, load tests, k8s |

Each stage has a journal in `notes/` with an answer key, what to observe, what to break, and what you found.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
claude                              # log in to Claude Code once; no API key needed
.venv/bin/python -m ssa.stage0_baseline damaged-and-late
.venv/bin/python -m pytest -q        # free: the answer key as unit tests for the refund calculator
```

LLM calls go through `src/ssa/llm.py`, which runs `claude -p` on your Claude Code login. Every stage calls
`complete()` and nothing else, so the backend can switch to the Anthropic API later without touching stage code.

## Layout

```
data/knowledge/   6 core policies + 20 help articles (some scoped with applies_to)
data/store.json   customers (tier, refund history), products, orders; "today" is fixed at 2026-10-16
data/evals/       scenarios.jsonl (7 customer messages with expected outcomes), retrieval.jsonl (policy questions),
                  retrieval_cases.jsonl (customer-style messages)
src/ssa/          the system as it grows
notes/            one learning journal per stage
runs/             raw output of every run (git-ignored), used to compare stages
var/              Stage 4's checkpoints and ledger databases (git-ignored)
```
