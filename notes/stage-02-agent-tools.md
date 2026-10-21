# Stage 2: a tool-using agent where code does the math

## What we built

Stage 2 exists in three versions. They share everything except the loop and the model, so their results are
directly comparable:

| Version | Loop | Model | Files |
|---|---|---|---|
| **LangGraph** (main) | a hand-drawn `StateGraph` | Gemini (`gemini-3.5-flash-lite`) | `agent_graph.py`, `stage2_langraph_agent.py` |
| `create_agent` | LangChain's prebuilt loop + middleware hooks | Gemini | `agent_prebuilt.py`, `stage2_prebuilt_agent.py` |
| CLI (the first one) | written by hand, JSON tool protocol | Claude via `claude -p` | `cli/` |

### The LangGraph version

```
START → prefetch → agent ──tool calls──→ tools ──→ agent
                     │ └─submit_final──→ validate ──rejected──→ agent
                     │                       └─accepted──→ END
                     ├─text only, no tool call──→ nudge ──→ agent
                     └─step budget used up──→ escalate → END
```

| Node | What it does | Why |
|---|---|---|
| `prefetch` | Plain code: looks up the order and customer on the message and puts them in the first prompt | In the CLI version every case's first step was those same lookups. A fixed step costs no LLM call |
| `agent` | One Gemini call with the tools bound (native function calling) | |
| `tools` | LangGraph's `ToolNode`. Tool exceptions become messages the model can react to | A wrong argument is something the model can fix |
| `validate` | The three gates on `submit_final` (below). A rejection goes back as that tool call's result | The final answer is just another tool, so the repair loop is an ordinary edge |
| `nudge` | "Call a tool, or finish with submit_final" | A text-only turn would otherwise stall the loop |
| `escalate` | Ends the case for a human after `MAX_STEPS` (6) LLM calls | Fail closed: no guessing when the budget runs out |

Tools get the case through `InjectedState`, so the model never sees or fills those parameters. A
`recursion_limit` on `invoke` backs up `MAX_STEPS` in case of a routing bug.

### Shared by all three versions

- **The model classifies, code calculates.** `schemas.py` has no field for an amount or an approval decision.
  The model reports case kinds and, per item, the issue and what the customer wants. `resolution.py` applies the
  rules: windows, the photo rule, tier rates and caps, approval and status. It reads dates, attachments and refund
  history from the database, not from the model.
- **Read-only tools** (`tools.py`): `get_order`, `get_customer`, `get_policy_sections` (by id from a table of
  contents in the system prompt), `search_policies` (the Stage 1b retriever), and `quote_resolution`, which shows
  the model what code will compute before it writes the reply.
- **Three gates on the final answer:**
  1. schema (pydantic)
  2. semantics: real ids that belong together (`check_decision`)
  3. grounding: every amount in the reply was looked up or quoted (`check_reply`)
- **Tests, no LLM:** `test_resolution.py` is the answer key as 12 calculator tests. `test_agent_graph.py` (6) and
  `test_agent_prebuilt.py` (5) drive each version with a scripted model through every route: reject then repair,
  accept, budget, tool error, text-only turn.
- **Automatic grading** on kind, status, approval, value and resolution lines (`stage2_langraph_agent.grade`).

### The `create_agent` version

The same behaviour, built from hooks on LangChain's prebuilt loop instead of drawn nodes:

| LangGraph (`agent_graph.py`) | `create_agent` (`agent_prebuilt.py`) |
|---|---|
| `prefetch` node | `before_agent` hook |
| `agent` ⇄ `tools` edges | the built-in loop |
| `submit_final` tool → `validate` node | `ToolStrategy(FinalAnswer)` + the gates in an `after_model` hook |
| rejected → edge back to `agent` | `after_model` returns `jump_to="model"` |
| `escalate` node | `ModelCallLimitMiddleware(run_limit=6)`; no answer counts as escalated |
| `nudge` node | `after_model` jump (needed; see finding 8) |

### The CLI version

The first version, before the switch to LangGraph. `claude -p` has no native tool use for our own functions, so
`cli/protocol.py` defines one: each step the model returns an `AgentStep` JSON (forced with `--json-schema`) that
either lists tool calls or gives the final answer. `cli/agent.py` runs the calls and re-sends the whole history to a
stateless call each step. That's what an API's tool use does under the hood, written out so you can see it.
`--json-schema` itself works through a hidden tool call, which is why such calls report two turns.

## Run it

```bash
.venv/bin/python -m pytest -q                            # calculator + both graphs with a scripted model, free
.venv/bin/python -m ssa.stage2_langraph_agent            # LangGraph + Gemini, all 7 scenarios
.venv/bin/python -m ssa.stage2_prebuilt_agent            # create_agent + Gemini
.venv/bin/python -m ssa.stage2_prebuilt_agent prompt-injection
```

The CLI version has no entry point now. Run it with
`main(names, runner=ssa.cli.agent.run_agent, tag="stage2-cli")` from `stage2_langraph_agent`.

## Findings (2026-10-19 to 2026-10-20)

### 1. Results so far

**CLI version on Claude (Opus): 7/7, graded by code instead of by reading.**

```
scenario                 kind stat appr value lines  steps calls  in tok  cached out tok  latency
damaged-and-late            ✓    ✓    ✓     ✓     ✓      3     3   18938   13206    1194    12.1s
laptop-cracked-screen       ✓    ✓    ✓     ✓     ✓      3     3   17963   13206    1076   13.65s
repeat-not-received         ✓    ✓    ✓     ✓     ✓      3     3   24041   21510    1232    30.4s
gold-late-delivery          ✓    ✓    ✓     ✓     ✓      4     4   30408   25035    1320   25.05s
change-of-mind-expired      ✓    ✓    ✓     ✓     ✓      3     3   17595   14907    1047   17.75s
shipping-question           ✓    ✓    ✓     ✓     ✓      3     3   16060   13206     648   11.45s
prompt-injection            ✓    ✓    ✓     ✓     ✓      3     3   17244   13206    1035    13.5s
```

**LangGraph version on Gemini: not yet run on the 7 scenarios.** The first attempt hit the free-tier limit
(finding 7). What we do have:
- **The 6 Stage 3 attacks against this graph: 4/6**, with 2 data leaks, 3–6 calls and 6–24s per case. Details are
  in `notes/stage-03-guardrails.md`, finding 1.
- **The Stage 3 graph, which is this graph plus guardrails, scored 7/7 on the scenarios** with
  `gemini-3.5-flash-lite`, in 3–5 calls each.

**`create_agent` version: tested only with the scripted model.** Its one real attempt hit the same quota limit.

Stages 0 to 1b were graded by a person reading free text. Now a regression shows up as a ✗ in a table, which is
what makes Stage 8's CI checks possible.

### 2. The table of contents fixed Stage 1b's vocabulary gap

For "the mug set inside was smashed", the agent asked for `refund-policy#damaged-or-wrong-items` straight away.
Mapping "smashed" to "damaged" is easy for a model reading a list of section titles, and was hard for an embedding
of the message. `search_policies` was never called. **For a small, well-titled corpus, letting the model pick from
a table of contents beats similarity search.** That stops scaling once the table of contents itself is too
big for the prompt (Stage 7).

### 3. Writing the rules as code found an ambiguity in the policy

The first test run failed: "standard shipping is refunded when every item in the order is refunded" read
literally includes change-of-mind returns. Code has to pick one reading (we chose store-fault refunds only; see
the comment in `resolution.py`). An LLM picks silently, maybe differently on each run. **Ambiguity in the
policy text becomes visible when it's turned into code.**

### 4. The grounding gate caught a bad reply

In a CLI smoke test, the model's first gold-late-delivery reply said "*our team is checking the amount*" for the
late credit, although `quote_resolution` had already returned USD 54.00. That would have told the customer about
a review that doesn't exist. `check_reply` rejected it, and the repaired reply was correct. In the full run every
final answer passed first time, so the repair loop is a safety net, not a crutch. Both graph versions have a
scripted test that replays this rejection.

### 5. The injection is handled by code paths, not by hoping the model refuses

The model set `suspicious: true` and code returned `quarantined` with zero lines. The reply was neutral, with no
card number and no amount (Stage 0 repeated "9911" back). Even if the model had been fooled, there's no field
through which it could have asked for $5,000 or a different card: amounts come from the order, and refunds only go
to the original payment method. The suspicious flag is still model judgement, which Stage 3 backs with checks in
code.

### 6. Cost, and why the LangGraph version has a prefetch node

CLI version on Claude:

| | Stage 0 | Stage 1b | Stage 2 (CLI) |
|---|---|---|---|
| Correct | 7/7 (read by hand) | 6/7 + 1 partial (by hand) | **7/7 (automatic)** |
| LLM calls per case | 1 | 1 | 3–4 |
| Input tokens per case | about 5.1k | about 4.2k | **16k–30k** (80% cached) |
| Output tokens per case | 600–2,000 | 560–2,100 | 650–1,300 |
| Latency per case | 6–27s | 6–21s | 11–30s |

- Input tokens grow with steps, because every step re-sends the system prompt and the whole history (quadratic
  in steps). Caching kept it affordable on Claude: 80% of input tokens were cache reads.
- Output is *shorter*: the model no longer writes out arithmetic, just a decision and a reply.
- Step 1 was the same in every case: `get_order` + `get_customer` + policy sections. **That's why the LangGraph
  version starts with `prefetch`**: plain code does those lookups, saving one LLM call per case. Use an agent loop
  where the path really varies, and plain code where it doesn't.

### 7. What changed with Gemini flash-lite

- **The free tier is too small for evals.** `gemini-3.6-flash` allows 20 requests per model per day, and one
  7-scenario run needs about 21. The first runs stopped with `429 RESOURCE_EXHAUSTED`, and the runner crashed on
  the first error, losing the batch. `stage3_eval.py` now records the error and stops cleanly.
- **The smaller model makes more schema mistakes.**
  - It sometimes wrapped the decision in a second `decision` field, which failed validation and cost a step.
  - For a question with no order it couldn't send `null` for `order_id`, and sent `"None"`, `"C-205"`,
    `"ORD-None"` and `"NONE"` until the budget ran out. Stage 3 sets `order_id` in code instead.
- **Latency varies a lot:** 4s to 64s for similar cases on the free tier.
- **Caching isn't measured yet.** The Claude CLI reported cache reads. `run_graph` collects Gemini's
  `cache_read` from the usage metadata, but no completed Stage 2 Gemini run has reported it, and `stage3_eval.py`
  doesn't record it.

### 8. LangGraph vs `create_agent`

- **`create_agent` needs less code for the loop:** 49 lines of hooks against 74 lines of graph wiring (97 vs 216
  in total, partly because it reuses the tools and prompt from `agent_graph.py`).
- **A text-only turn ends the prebuilt loop.** `create_agent` treats a turn with no tool call as "finished", so
  the case escalated after one call. It needed an `after_model` jump to nudge the model on, as the graph does.
- **The repair history contradicts itself.** The framework has already answered the final-answer call with
  *"Returning structured response…"* before our check rejects it, so the model sees "accepted", then "rejected".
  The graph answers that call itself, so its history stays consistent.
- **The repair path is hidden.** In the graph it's an edge you can draw and test; in `create_agent` it's a
  `jump_to` inside a hook.
- Takeaway: `create_agent` is fine while the agent really is one loop. LangGraph was the better fit here as soon
  as a fixed step (prefetch) and a gate with its own route (validate) appeared, and Stage 3 builds on it.

## Break it

- Remove the photos from `laptop-cracked-screen`. Status should be `needs_info`, with a reply asking for photos.
- Set `MAX_STEPS = 2` in `agent_graph.py`. Which cases escalate, and is "escalated" the right failure?
- Comment out `check_reply` in the `validate` node and run `gold-late-delivery` 5 times. How often is the reply
  wrong about money?
- Remove the `nudge` jump from `agent_prebuilt.py` and run `test_a_text_only_turn_is_nudged`.
- **Rule drift:** change the Gold cap to 25% in `loyalty-tiers.md` only. The doc and the code now disagree, and
  only the code is used for the amount.
- The indirect-injection and data-leak attacks that were listed here are now in `data/evals/attacks.jsonl`
  (Stage 3).

## Bottlenecks, and where they get fixed

| Bottleneck | Fixed in |
|---|---|
| Tools return any customer's data; injected text in tool results is trusted | Stage 3 (done): case-scoped tools, sanitised tool data |
| `suspicious` is still the model's call | Stage 3 (done): intake checks in code quarantine with no LLM call |
| The model fills fields code already knows (`order_id`) | Stage 3 (done): set by code |
| A crash mid-loop loses the work; approval has nowhere to wait | Stage 4: checkpoints, approval queue |
| 3–4 sequential LLM calls, 4–64s per case, blocking; free-tier quota | Stage 5: async workers; Stage 9: fallback models, retries |
| Rules live in both docs and code and can drift | Version them together; a test per rule that cites its section |
