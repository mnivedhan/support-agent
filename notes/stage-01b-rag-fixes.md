# Stage 1b: fixing the RAG bottlenecks from Stage 1

## What we built

Each fix can be switched on separately, so the eval measures what each one adds (an ablation ladder,
`rag/retriever.py`):

| Rung | Fix | Code |
|---|---|---|
| `stage1` | Stage 1 as it was | |
| `+heading` | One chunk per markdown section, titled `"<doc title> > <heading>"` | `rag/chunking.py` |
| `+hybrid` | Vector + BM25 keyword search, fused by rank (RRF) | `rag/bm25.py`, `rag/index.py` |
| `+context` | Code looks up the DB and writes each notable fact as a sentence; each fact is its own query | `case_context.py` |
| `+filter` | Scoped articles (`applies_to: marketplace_item`, ...) dropped when the order doesn't match | front matter, `retriever.py` |
| `+pin-sop` | The SOP always goes in the prompt whole, never retrieved | `retriever.py` |

Also:
- `data/evals/retrieval_cases.jsonl`: 12 customer-style messages ("never got my headphones. refund.") with
  customer and order ids. Together with the 7 scenarios, that's 19 cases.
- `rag_eval.py` now reports **covered** (every required doc present), recall, precision, wrong-scope docs, tokens
  and latency for every rung, on both the customer cases and Stage 1's policy questions.
- `stage1b_rag.py`: end to end with the last rung. The retrieved sections go in the **user message**; see finding 6.

## Run it

```bash
.venv/bin/python -m ssa.rag_eval              # the ladder, free
.venv/bin/python -m ssa.rag_eval +pin-sop     # plus per-case detail for one rung
.venv/bin/python -m ssa.stage1b_rag           # end to end, 7 LLM calls
```

## Findings (2026-10-18)

### 1. The ladder

```
                  ---------- customer cases (19) ----------           -- questions (18) --
strategy    covered  recall  precision  wrong-scope  tokens     ms    covered  recall
stage1         0.53    0.73       0.43         0.47     362     19       0.94    0.97
+heading       0.47    0.68       0.36         0.79     346     15       0.89    0.92
+hybrid        0.42    0.71       0.38         0.74     337     13       0.89    0.92
+context       0.58    0.82       0.34         0.95     529     44       0.89    0.92
+filter        0.68    0.86       0.38         0.00     538     38       0.89    0.92
+pin-sop       0.84    0.94       0.35         0.00    1057     34       0.94    0.94
```

**Covered went from 0.53 to 0.84 on real customer messages.** The policy-question set barely moved
(0.94 → 0.94). Stage 1's eval couldn't have shown either problem or fix.

### 2. Heading chunks made the score *worse*, and the score was the problem

Stage 1's fixed chunks of the SOP were long, generic blobs that matched almost anything. Because a doc counts as
"found" if *any* of its chunks is retrieved, those lucky hits looked like successes. Heading chunks made each SOP
section specific, so the luck stopped. **Doc-level labels give credit for the wrong section.** See finding 5 for
the same problem reaching the LLM.

### 3. BM25 helps exact terms and hurts vague messages

- It helps: "Photos attached" → BM25 matched "photo evidence" → `fraud-and-abuse` was found for the laptop.
- It hurts: "Something in order 1042 **broke** during **delivery**". No doc says "broke" (they say "damaged"), so
  common shop words like "order" and "delivery" win. RRF gives that noise the same weight as the vector ranking.
- **Tuning experiment:** stopwords + light stemming raised the policy questions to 1.00 but dropped the customer
  cases at `+hybrid` from 0.42 to 0.32. Not kept. Tuning against one eval set can hurt real traffic.
- With 19 cases, one case flipping moves "covered" by 0.05. **Differences under about 0.1 are mostly noise.**
  Production evals need hundreds of cases.

### 4. Case context and filters did the heavy lifting

- `+context` added the facts the message never states: "The customer is a Gold loyalty tier member", "2 refunds or
  replacements in the last 90 days", "card ending 9911, which is not the card used". `loyalty-tiers` and
  `fraud-and-abuse` now show up where they're needed.
- More queries = more noise: wrong-scope docs rose to 0.95 per case (the partner card's "Gold" tier came along
  with the loyalty Gold fact). `+filter` removed every one of them (0.95 → 0.00). The filter is skipped when there's
  no order, so "do you do warehouse pickup?" still finds the pickup article.
- Pinning the SOP fixed most of the rest, and it's not cheating: an operating procedure is needed on every case.
  **Not everything should be retrieved.** Small, always-relevant docs belong in the prompt.
- Cost: context tokens about tripled (362 → 1,057) and retrieval latency doubled (19 → 34 ms, one embedding per
  extra query). Precision stayed low (about 0.35), so most of what's in context still isn't needed.

### 5. End to end: 6 of 7 right, 1 partial

| Scenario | Stage 1 | Stage 1b |
|---|---|---|
| damaged-and-late | not run | ⚠️ right numbers ($51.25), but held the $40 refund |
| laptop-cracked-screen | not run | ✅ $899, approval |
| repeat-not-received | ❌ | ✅ $70.99, approval (3rd in 90 days) |
| gold-late-delivery | ❌ | ✅ $68.99 |
| change-of-mind-expired | not run | ✅ denied, cited the warranty colour exclusion |
| shipping-question | not run | ✅ |
| prompt-injection | not run | ✅ flagged, nothing refunded |

The partial is finding 2 again: the eval said `refund-policy` was retrieved, but the section was *Change of mind*,
not *Damaged or wrong items*. The model said so: "the damaged-item section of the refund policy was not
provided". The vocabulary gap is the cause: "smashed" / "in pieces" / "broke" don't match "Damaged or wrong items"
strongly enough.

### 6. Prompt layout decides whether caching works

- Stage 0 sends an identical prompt every call: about 7.5k tokens read from cache per call.
- Stage 1b first had the retrieved chunks inside the system prompt: **0 cached**. Moving the static blocks
  ahead of the chunks didn't help either: still 0.
- The cache only matches at **breakpoints**, and the CLI places one at the end of the system prompt. Any
  per-request content in the system prompt means that breakpoint never matches.
- Fix: retrieved sections go in the user message. Result: 2,773 of 4,378 input tokens (63%) cached from the second
  call on.
- **Rule: static content (instructions, pinned docs, reference data) in the system prompt, per-request content
  in the user turn.**

### 7. Your experiment (saved as `git stash list`, stash@{0})

You tried storing a vector on each `Chunk` and embedding chunk by chunk. Measured:
- One batched embedding call for 47 chunks: 0.91s; one call per chunk: 1.05s. Batching barely matters against a
  local Ollama, but against a hosted API it's 47 round-trips and 47 rate-limit hits.
- `embed_query(query)` inside the per-chunk loop embeds the same query 47 times: **0.65s per search instead of
  14ms**, and it grows with the corpus. Embed the query once, and keep vectors in one matrix so a single `@`
  scores them all.

## Break it

- `python -m ssa.rag_eval +filter`: which cases still miss without the pinned SOP, and why?
- Change `FACT_K` to 1 and `MESSAGE_K` to 2 in `retriever.py`. How much coverage do you lose per token saved?
- In `HybridIndex.scores`, weight the vector ranking 2× the BM25 one. Does "broke during delivery" recover?
- Add `"seller": "acme-gadgets"` to `P-HDP` in `store.json`. Does `marketplace-seller-returns` now appear for
  order 1090, and does the filter still hide it from order 1042?
- Delete the `if customer["tier"] != "standard"` condition in `case_context.py` and watch precision.

## Bottlenecks left, and where they get fixed

| Bottleneck | Fix |
|---|---|
| Customer words ≠ policy words ("in pieces" vs "damaged") | Classify the case first, then retrieve by case type (Stage 2: the model asks for policies with tools) |
| Doc-level labels give credit for the wrong section | Label required **sections** (`refund-policy#damaged`), score at that level |
| Hand-written fact templates; every new fact needs code | Stage 2: tools the model calls (`get_order`, `get_refund_history`) |
| Precision about 0.35: most context isn't needed | Reranking, score thresholds (Stage 7) |
| The whole store DB is still in every prompt | Stage 2: look up only this order and customer |
| LLM still does the arithmetic | Stage 2: code computes, the model explains |
