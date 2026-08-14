#!/usr/bin/env python3
"""
Render the JSON array returned by ChatGPT/Codex into a terminal table.

Usage:
    python render_ai_review.py ai_review.json
"""
import argparse
import json
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("json_file")
    args = p.parse_args()

    with open(args.json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for x in data:
        rows.append({
            "Ticker": x.get("ticker"),
            "Company": x.get("company"),
            "Market": x.get("market"),
            "Signal date": x.get("signal_date"),
            "Support %": x.get("evidence_support_pct"),
            "Classification": x.get("classification"),
            "Key positive": x.get("key_positive"),
            "Key risk": x.get("key_risk"),
        })

    df = pd.DataFrame(rows).sort_values("Support %", ascending=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
