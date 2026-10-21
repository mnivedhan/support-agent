"""The side effects: money moved and emails sent. Stands in for the payment provider and the email service.

Every write has an idempotency key, and a second write with the same key is a no-op. That's what makes it safe to
re-run a node. LangGraph saves a checkpoint *after* a node finishes, so a crash between "refund sent" and
"checkpoint saved" re-runs the node on recovery. Without the key, that customer is refunded twice
(tests/test_stage4_graph.py::test_crash_after_paying_does_not_pay_twice).

Real payment APIs work the same way: Stripe takes an Idempotency-Key header. The key has to come from the case,
not from the attempt: case id + line, never a timestamp or a random id.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

SCHEMA = """
create table if not exists actions (
    key text primary key, case_id text not null, line text not null, approved_by text not null, at text not null);
create table if not exists outbox (
    key text primary key, case_id text not null, kind text not null, body text not null, at text not null);
"""


class Ledger:
    def __init__(self, path: str | Path = ":memory:"):
        self.db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self.db.executescript(SCHEMA)

    def execute(self, case_id: str, lines: list[str], approved_by: str) -> list[str]:
        """Carry out each resolution line once. Returns the lines that ran now; repeats are skipped."""
        done = []
        for i, line in enumerate(lines):
            cur = self.db.execute("insert or ignore into actions values (?, ?, ?, ?, ?)",
                                  (f"{case_id}:{i}:{line}", case_id, line, approved_by, _now()))
            if cur.rowcount:
                done.append(line)
        return done

    def send(self, case_id: str, kind: str, body: str) -> bool:
        """Send one email of each kind per case (ack, confirm, reject). False if it was already sent."""
        cur = self.db.execute("insert or ignore into outbox values (?, ?, ?, ?, ?)",
                              (f"{case_id}:{kind}", case_id, kind, body, _now()))
        return bool(cur.rowcount)

    def actions(self, case_id: str | None = None) -> list[dict]:
        return self._rows("actions", case_id)

    def emails(self, case_id: str | None = None) -> list[dict]:
        return self._rows("outbox", case_id)

    def _rows(self, table: str, case_id: str | None) -> list[dict]:
        cur = self.db.execute(f"select * from {table}" + (" where case_id = ?" if case_id else "") + " order by at",
                              (case_id,) if case_id else ())
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
