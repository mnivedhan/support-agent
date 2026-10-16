"""Loaders for the shared data/ folder. Every stage reads the same inputs so results stay comparable."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_knowledge() -> dict[str, str]:
    """Return {doc_id: markdown} for every policy and SOP."""
    return {p.stem: p.read_text() for p in sorted((DATA_DIR / "knowledge").glob("*.md"))}


def load_store() -> dict:
    """Customers, products and orders: the store's database snapshot."""
    return json.loads((DATA_DIR / "store.json").read_text())


def load_scenarios() -> list[dict]:
    lines = (DATA_DIR / "evals" / "scenarios.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]
