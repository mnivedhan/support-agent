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
| 1 | Simple RAG: fixed-size chunks, embeddings, cosine search | Misses exact terms like "Gold" or order IDs; splits rules at bad places → hit@k evals, BM25 + dense search, RRF, reranking |
| 2 | Structured output + read-only store tools; code does the math | LLM arithmetic and date errors, invalid JSON → pydantic checks, retries, a refund calculator in code |
| 3 | LangGraph state machine with a critic step | Endless loops, prompt injection → step budgets, guardrails, least-privilege tools |
| 4 | Manager approval + saved checkpoints | Process crashes while waiting for approval → state in a database, resume on any worker |
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
```

LLM calls go through `src/ssa/llm.py`, which runs `claude -p` on your Claude Code login. Every stage calls
`complete()` and nothing else, so the backend can switch to the Anthropic API later without touching stage code.

## Layout

```
data/knowledge/   6 policies: support SOP, refunds, shipping, loyalty tiers, warranty, fraud
data/store.json   customers (tier, refund history), products, orders; "today" is fixed at 2026-10-16
data/evals/       scenarios.jsonl (7 customer messages with expected outcomes), retrieval.jsonl (for Stage 1)
src/ssa/          the system as it grows
notes/            one learning journal per stage
runs/             raw output of every run (git-ignored), used to compare stages
```
