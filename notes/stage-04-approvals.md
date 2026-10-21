# Stage 4: manager approval and checkpoints that survive a crash

Stage 3 ends a case at `awaiting_approval` or `escalated` and then forgets it. The customer has been told "a
manager is reviewing this", but there's no manager, no queue, and nothing that remembers the case after the
process exits. Stage 4 makes a case durable. It can wait days for a human, survive a crash, resume on any worker,
and carry out the decision exactly once.

## What we built

```
START → handle ──auto-executable──→ execute ──→ confirm → END
          │                           │ └ data changed since the decision ──→ review
          └→ acknowledge ──needs a human──→ review ──approve──→ execute
                 │                             └─reject or close──→ close → END
                 └ otherwise (denied, needs info, question) → END
```

| File | What |
|---|---|
| `stage4_graph.py` | the graph, plus the queue API: `submit`, `pending`, `decide`, `stuck`, `recover` |
| `ledger.py` | the side effects (payments, emails) behind idempotency keys; stands in for Stripe and an email service |
| `stage4_queue.py` | a CLI where every command is a separate process: workers, the reviewer, recovery |
| `tests/test_stage4_graph.py` | 13 tests: restarts, crashes, stale data, double clicks, bad answers. Scripted model, free |

| Node | Does | Why |
|---|---|---|
| `handle` | runs the whole Stage 3 graph as one step, keeps only the outcome | the checkpoint holds the case, not the conversation (finding 2) |
| `acknowledge` | first email: the agent's reply, or a holding reply if escalated | the customer hears something before the case waits |
| `review` | `interrupt(request)`: saves and stops; `Command(resume=answer)` continues it | the human is just another step, and the process can exit while it waits |
| `execute` | re-runs the calculator on today's data; pays through the ledger | approvals go stale (finding 4); crashes repeat nodes (finding 3) |
| `confirm` / `close` | last email; after a human decision code writes it from the executed lines | can't promise anything that wasn't approved |

The store, the model and the ledger come in through LangGraph's runtime **context**, not the state. They are not
saved in checkpoints, and every resume reads today's data.

Reviewer actions: `approve` / `reject` when there's something to approve; only `close` when there isn't
(quarantined, budget used up, nothing left to do). An invalid answer (no reviewer name, an action not offered)
is asked again inside the node and moves nothing.

## Run it

```bash
.venv/bin/python -m pytest -q                                   # 55 tests, free
.venv/bin/python -m ssa.stage4_queue run all                    # the agent on the 7 scenarios (Gemini)
.venv/bin/python -m ssa.stage4_queue queue                      # what's waiting, and why
.venv/bin/python -m ssa.stage4_queue approve laptop-cracked-screen --by ana
.venv/bin/python -m ssa.stage4_queue ledger                     # money moved and emails sent
rm -rf var                                                      # start over
```

## Findings (2026-10-20)

### 1. Break it: in-memory state forgets the case, after the customer was told to wait

`run laptop-cracked-screen --memory` (the default `InMemorySaver`):

```
[laptop-cracked-screen]  handled: 3 LLM calls, ["replace P-LAP"]  → sent ack → waiting_for_review
queue: 1 waiting                                       ← same process
$ python -m ssa.stage4_queue queue --memory
queue: 0 waiting (empty)                               ← next process
```

3 LLM calls spent, the customer told "awaiting a manager's approval", and the case is gone. Any deploy, crash or
scale-down does this to every case in the queue. Fix: `SqliteSaver` on `var/checkpoints.db`. The same commands
then show the case in a new process, and the manager approves it there with **0 LLM calls**.

### 2. The naive checkpointer: 144 KB per simple case, and a hidden second one

Adding a checkpointer to the Stage 3 graph as it is (one damaged-mug case, 4 LLM calls, scripted):

| Setup | Checkpoints | Bytes |
|---|---|---|
| Stage 3 graph + `SqliteSaver` | 11 | 143,955 |
| Stage 4, first try | 16 | 154,253 |
| Stage 4, inner graph with `checkpointer=False` | **5** | **10,292** |

- Every checkpoint copies the whole state, including the full message history (system prompt, policy text, tool
  results), so total size grows with steps × history. The store is in the Stage 3 state too, though here it's
  only 3 KB.
- The first Stage 4 version was *bigger*, with a tiny state of its own. Grouping by `checkpoint_ns` showed why:
  4 checkpoints (6 KB) for the outer graph and 7 (64 KB) under `handle:<id>`. **A graph invoked inside a node
  inherits the parent's checkpointer.** `compile(checkpointer=None)` means "inherit", not "none"; turning it off
  takes `checkpointer=False`. There's a regression test for it now.
- The tradeoff is **checkpoint granularity = what you redo after a crash**. A crash inside the agent loop re-runs
  the whole loop: a few LLM calls, safe because every tool is read-only. A crash after paying must never redo the
  payment. So the expensive-but-safe part gets one coarse checkpoint, and the side effects get their own nodes.

### 3. Every node can run twice, so every side effect needs an idempotency key

Two ways a node runs again:
- **Resume restarts the interrupted node from its first line.** `review` runs once when the case arrives and
  again when the manager answers. Anything before `interrupt()` must be free of side effects
  (`test_interrupted_node_reruns_from_its_first_line`).
- **A checkpoint is saved after a node finishes.** If the worker dies between "refund sent" and "checkpoint
  saved", recovery runs `execute` again.

Run for real, across processes:

```
$ approve laptop-cracked-screen --by ana --crash-at execute
SimulatedCrash: worker died after paying, before the checkpoint was saved
$ ledger            → refund P-LAP 899.00   approved by ana          (paid)
$ recover laptop-cracked-screen
  executed: lines ["refund P-LAP 899.00"], ran_now: []            ← ran again, paid nothing
  sent confirmation → resolved
$ ledger            → one refund
```

The ledger's key is `case_id:line_index:line`: built from the case, never from the attempt (no timestamps, no
random ids). Payment APIs take the same thing as an `Idempotency-Key` header. Emails are keyed `case_id:kind`, so
a recovered case doesn't send the confirmation twice.

A second click on approve is refused (`isn't waiting for a review`). That check is for the reviewer; two clicks
racing could both pass it. The ledger key is what actually stops the second payment.

### 4. Approvals go stale: same value, different action

The laptop case waited for approval as `replace P-LAP` (USD 899). While it waited, the last laptop sold
(`--set products.P-LAP.stock=0`). The manager clicked approve:

```
  reviewed: approve by ana, approved ["replace P-LAP"]
  stale: was ["replace P-LAP"] → now ["refund P-LAP 899.00"]
queue: 1 waiting
    proposal: ['refund P-LAP 899.00'], USD 899.00
    why: data changed since the last decision: ['replace P-LAP'] -> ['refund P-LAP 899.00']
```

Without the re-check, `execute` would have shipped a laptop that doesn't exist. **The case value didn't change**
(899 either way), so comparing amounts alone would have missed it. The check compares the lines. An approval
is for a specific action on specific data, and it expires when either changes. The same applies to anything
slow: refund history (a 3rd claim filed elsewhere meanwhile), delivery status, a policy change.

### 5. The real crash: the LLM provider, not our code

Running the 7 scenarios on Gemini (`gemini-3.5-flash-lite`):

| Case | Result |
|---|---|
| damaged-and-late | handled (4 calls), auto-executed: USD 40.00 refund + 11.25 credit, confirmed |
| laptop-cracked-screen | handled (3 calls), queued; the manager side above ran on it |
| repeat-not-received | **stuck at `handle`**: Gemini hung about 10 minutes, then `Connection reset by peer`. Twice (the second via `recover`) |
| gold-late-delivery | **stuck at `handle`**: hung over 10 minutes; the run was stopped |
| the other 3 | not started |

- The first hang killed the whole batch at 13m47s. The runner now catches a failed case, leaves its checkpoint
  at `next: handle`, and carries on. `stuck` lists such cases and `recover` re-runs them.
- **Nothing was paid or sent for the stuck cases.** They died before any side effect, which is the outcome
  the design was meant to guarantee.
- A direct one-line call with `timeout=30, max_retries=0` hung for over 2 minutes as well, while the endpoint
  answered HTTP in 0.6s. So the timeout didn't take effect through `init_chat_model`; not investigated yet.
- One worker blocked for 10+ minutes on one call is the bottleneck of the next stages: Stage 5 moves the work to
  background workers, and Stage 9 adds working timeouts, retries with backoff, and a fallback model.

The remaining 5 scenarios still need a Gemini run once the API responds: `python -m ssa.stage4_queue recover`
then `run all`.

## Break it

- Move `ledger.send(... "ack" ...)` into `review`, above the `interrupt()`. Then drop the outbox key. How many
  acks does one approved case send?
- Make the ledger key `f"{case_id}:{time.time()}"` and rerun the crash test.
- Compare only `case_value` in `execute`'s stale check and rerun the out-of-stock test.
- Two terminals: `approve` the same case at the same moment. Who wins, and how many refunds are there?
- Change the graph (rename a node) while a case is waiting, then approve it. What happens to old checkpoints
  after a deploy?

## Bottlenecks left, and where they get fixed

| Bottleneck | Fix |
|---|---|
| A hung LLM call blocks the worker for 10+ minutes; a batch runs cases one by one | Stage 5: API returns 202, background workers; Stage 9: working timeouts, retries, fallback model |
| `pending()` scans every checkpoint of every thread | Stage 5: a cases table with a status column and an index |
| The same complaint sent twice under two case ids is two cases (keys are per case) | Stage 6: dedupe at intake by message id; idempotent consumers |
| SQLite: one writer, one machine | Postgres checkpointer (`langgraph-checkpoint-postgres`) when workers run on several machines |
| Graph changes while cases are waiting | Version the graph; drain or migrate old threads (Stage 9) |
| No one is told a case is waiting; no SLA on reviews | Notifications and review-time metrics (Stage 8) |
