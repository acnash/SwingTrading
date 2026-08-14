#!/usr/bin/env python3
"""Validate three independent AI reviews and render their consensus scores."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd


REVIEW_COUNT = 3


def classification_for_score(score: int) -> str:
    if score >= 80:
        return "STRONGLY SUPPORTS BUY SIGNAL"
    if score >= 65:
        return "SUPPORTS BUY SIGNAL"
    if score >= 50:
        return "NEUTRAL / MIXED"
    if score >= 35:
        return "WEAKENS BUY SIGNAL"
    return "STRONGLY WEAKENS BUY SIGNAL"


def research_action_for_score(score: int) -> str:
    if score >= 65:
        return "BUY CANDIDATE"
    if score >= 50:
        return "WATCH / NO TRADE"
    return "REJECT BUY / REVIEW EXIT"


def load_review(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path}: top-level JSON must be an array")

    review: dict[str, dict] = {}
    for item in data:
        if not isinstance(item, dict) or not item.get("ticker"):
            raise ValueError(f"{path}: every review row must contain a ticker")
        ticker = str(item["ticker"])
        if ticker in review:
            raise ValueError(f"{path}: duplicate ticker {ticker}")
        score = item.get("evidence_support_pct")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 100:
            raise ValueError(f"{path}: {ticker} has an invalid evidence_support_pct")
        review[ticker] = item
    return review


def aggregate_reviews(paths: list[Path]) -> pd.DataFrame:
    if len(paths) != REVIEW_COUNT:
        raise ValueError(f"Exactly {REVIEW_COUNT} independent review files are required")

    reviews = [load_review(path) for path in paths]
    expected = set(reviews[0])
    for path, review in zip(paths[1:], reviews[1:]):
        if set(review) != expected:
            missing = sorted(expected - set(review))
            extra = sorted(set(review) - expected)
            raise ValueError(f"{path}: ticker set differs; missing={missing}, extra={extra}")

    rows = []
    for ticker in expected:
        items = [review[ticker] for review in reviews]
        scores = [float(item["evidence_support_pct"]) for item in items]
        mean = sum(scores) / REVIEW_COUNT
        rounded_mean = int(Decimal(str(mean)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        first = items[0]
        rows.append({
            "Ticker": ticker,
            "Company": first.get("company", ""),
            "Market": first.get("market", ""),
            "Signal date": first.get("signal_date", ""),
            "Review 1 %": scores[0],
            "Review 2 %": scores[1],
            "Review 3 %": scores[2],
            "Mean %": round(mean, 2),
            "Range": round(max(scores) - min(scores), 2),
            "Classification": classification_for_score(rounded_mean),
            "Research action": research_action_for_score(rounded_mean),
        })

    return pd.DataFrame(rows).sort_values(
        ["Mean %", "Ticker"], ascending=[False, True], ignore_index=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate exactly three independent fundamental reviews."
    )
    parser.add_argument("json_files", nargs=3, type=Path, metavar="REVIEW_JSON")
    parser.add_argument("--output-csv", type=Path, help="Optional consensus CSV path")
    args = parser.parse_args()

    try:
        consensus = aggregate_reviews(args.json_files)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        consensus.to_csv(args.output_csv, index=False)
    print(consensus.to_string(index=False))
    print("\nResearch actions combine the three fundamental scores; they are not trade advice.")


if __name__ == "__main__":
    main()
