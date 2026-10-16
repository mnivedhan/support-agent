# Stage 1: simple RAG

## What we built

- **A bigger knowledge base.** Added 20 help-centre articles: 26 docs, about 2,500 words. Some are traps that use
  the same words in a different scope:
  - `marketplace-seller-returns`: a 14-day window, but only for third-party sellers
  - `partner-credit-card`: has a "Gold" tier unrelated to loyalty Gold
  - `holiday-returns`, `bulk-orders`, `subscription-coffee`, `store-pickup`: exceptions to the late-credit and
    returns rules
- `rag/chunking.py`: **fixed-size** 400-character chunks, 50 characters of overlap. Ignores headings.
- `rag/embeddings.py`: `nomic-embed-text` on local Ollama (free). Note the required `search_document:` /
  `search_query:` prefixes.
- `rag/index.py`: numpy brute-force cosine search, rebuilt on every start (47 chunks in about 1.4s).
- `rag_eval.py`: hit@k, recall@k and MRR over `data/evals/retrieval.jsonl`. Free, with no LLM involved.
- `stage1_rag.py`: Stage 0 with the top-4 chunks for the customer's message instead of every policy. The store DB
  still goes in whole.
- `scenarios.jsonl` now lists `required_docs` per scenario, so every run reports which policies were missing.

## Run it

```bash
.venv/bin/python -m ssa.rag_eval          # retrieval only, free
.venv/bin/python -m ssa.stage1_rag        # end to end, 7 LLM calls
```

## Findings (2026-10-16)

**1. The retrieval eval said "great" and production said "broken".**
- `rag_eval`: hit@4 **1.00**, recall@4 0.97, MRR 0.88. Looks ready to ship.
- On the 7 real customer messages, a required policy was **missing in 5 of 7**:

| Scenario | Missing | Why |
|---|---|---|
| damaged-and-late | shipping-policy | The message says "late" once; "smashed mugs" dominates the vector |
| laptop-cracked-screen | fraud-and-abuse | The photo rule lives in the fraud doc; the message never says fraud |
| repeat-not-received | shipping, fraud, support-sop | Got `marketplace-seller-returns`, `warranty` instead |
| gold-late-delivery | loyalty-tiers | **The message never says "Gold"**. The tier is in the DB, not the text |
| prompt-injection | support-sop | Got `account-security`, `partner-credit-card` |

The eval questions were written by someone who knew the answer, in policy language ("What late credit does a
Gold customer get?"). Customers don't write like that. **An eval set that doesn't look like production traffic
measures nothing.**

**2. The query is missing information that only exists in the database.** Whether the Gold rule or the
repeat-claims rule applies depends on the customer's tier and refund history. Neither is in the message, so no
embedding model can retrieve the right rule from the message text alone.

**3. Fixed-size chunks cut a rule in half.** The shipping policy was split mid-rule: "5 percent of the subtotal
for each day late" ends chunk #1, and chunk #2 starts with "capped at 20 percent" (via the overlap). Only chunk #2
was retrieved. The model replied: *"The per-day rate is cut off. I can't calculate the late-delivery credit."*

**4. The traps worked.** `marketplace-seller-returns` was retrieved in 5 of 7 scenarios, even though no order
involves a marketplace seller.

**5. What the LLM did with missing context** (`gold-late-delivery`, `repeat-not-received`):
- It **didn't make up numbers**. It said what was missing and refused to finalise the late credit or the approval
  decision. That's good model behaviour, but a weaker model or a different prompt would happily guess $50.99
  (the standard rate) instead of $68.99.
- On repeat-not-received it escalated anyway, "because no policy covers the claim": the right outcome for the
  wrong reason.
- **0 of 2 correct**, where Stage 0 got 2 of 2.

**6. Tokens:** about 3,200 input tokens per call vs about 5,200 for Stage 0 on the same 26-doc corpus. That saves
about 40% of the prompt but loses correctness. At this corpus size, RAG is not worth it yet. It earns its place
once the corpus is too big or too noisy to put in the prompt (Stage 7).

## Break it

- `python -m ssa.rag_eval 8`: does k=8 fix the misses? What does it cost in tokens and noise?
- Change `size=400` to 1000 in `build_index`. Do the cut-rule problems go away, and what happens to precision?
- Remove the `search_query:` / `search_document:` prefixes in `embeddings.py` and re-run `rag_eval`.
- Add 5 retrieval queries written the way customers write ("my stuff never came!!") and watch the scores drop.

## Bottlenecks, and the fixes coming in Stage 1b

| Bottleneck | Fix |
|---|---|
| The eval doesn't look like production traffic | Build the retrieval eval from real messages (`required_docs` per scenario) |
| The query lacks facts from the DB (tier, history, case type) | **Build the query from case context**: classify the case, look up the order and customer, then retrieve per need |
| Rules cut in half | **Heading-aware chunking**: one section per chunk, with the doc title prepended |
| Similar words in the wrong scope | **Metadata filters** (doc_type, scope) + hybrid **BM25** for exact terms + **reranking** |
| Always top-k, even when a doc is irrelevant | Score thresholds; retrieve per required rule, not one blob |
