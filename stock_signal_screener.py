#!/usr/bin/env python3
"""
Daily moving-average crossover screener for US + UK equities.

Default universes:
  - S&P 500
  - FTSE 100

Signal definition (daily candles):
  BUY  = short SMA was <= medium SMA yesterday and is > medium SMA today.
  SELL = short SMA was >= medium SMA yesterday and is < medium SMA today.

Default windows:
  short  = 20 trading days
  medium = 50 trading days

Outputs:
  output/all_signals.csv
  output/buy_signals.csv
  output/buy_signal_review_prompt.txt

This is a research/screening tool, not investment advice.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


def normalise_us_symbol(symbol: str) -> str:
    # Yahoo uses BRK-B rather than BRK.B, etc.
    return str(symbol).strip().replace(".", "-")


def normalise_uk_symbol(symbol: str) -> str:
    s = str(symbol).strip()
    # Common table variants.
    s = s.replace("LON:", "").replace("LSE:", "").strip()
    if not s.endswith(".L"):
        s += ".L"
    return s


def get_sp500() -> pd.DataFrame:
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = tables[0][["Symbol", "Security"]].copy()
    df.columns = ["ticker", "company"]
    df["ticker"] = df["ticker"].map(normalise_us_symbol)
    df["market"] = "US"
    return df


def get_ftse100() -> pd.DataFrame:
    tables = pd.read_html("https://en.wikipedia.org/wiki/FTSE_100_Index")
    candidate = None
    for table in tables:
        cols = {str(c).lower(): c for c in table.columns}
        if "ticker" in cols and ("company" in cols or "constituent" in cols):
            candidate = table
            break
    if candidate is None:
        # Fallback: find a table with an EPIC/Ticker-like field.
        for table in tables:
            lowered = [str(c).lower() for c in table.columns]
            if any("ticker" in c or "epic" in c for c in lowered):
                candidate = table
                break
    if candidate is None:
        raise RuntimeError("Could not locate FTSE 100 constituent table.")

    ticker_col = next(
        c for c in candidate.columns
        if "ticker" in str(c).lower() or "epic" in str(c).lower()
    )
    company_col = next(
        (c for c in candidate.columns
         if "company" in str(c).lower() or "constituent" in str(c).lower()),
        candidate.columns[0]
    )

    df = candidate[[ticker_col, company_col]].copy()
    df.columns = ["ticker", "company"]
    df["ticker"] = df["ticker"].map(normalise_uk_symbol)
    df["market"] = "UK"
    return df


def load_custom_csv(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["ticker", "company", "market"])
    df = pd.read_csv(path)
    required = {"ticker", "company", "market"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Custom CSV is missing columns: {sorted(missing)}")
    return df[["ticker", "company", "market"]].copy()


def build_universe(include_sp500: bool, include_ftse100: bool, custom_csv: str | None) -> pd.DataFrame:
    parts = []
    if include_sp500:
        parts.append(get_sp500())
    if include_ftse100:
        parts.append(get_ftse100())
    custom = load_custom_csv(custom_csv)
    if not custom.empty:
        parts.append(custom)

    if not parts:
        raise ValueError("No stock universe selected.")

    universe = pd.concat(parts, ignore_index=True)
    universe = universe.drop_duplicates("ticker").sort_values(["market", "ticker"])
    return universe


def extract_one_ticker(downloaded: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    yfinance returns different shapes for one versus many tickers.
    Convert either form into a simple OHLCV dataframe.
    """
    if downloaded.empty:
        return pd.DataFrame()

    if isinstance(downloaded.columns, pd.MultiIndex):
        # Newer yfinance commonly returns Price x Ticker.
        lvl0 = downloaded.columns.get_level_values(0)
        lvl1 = downloaded.columns.get_level_values(1)

        if ticker in lvl1:
            out = downloaded.xs(ticker, axis=1, level=1, drop_level=True).copy()
        elif ticker in lvl0:
            out = downloaded.xs(ticker, axis=1, level=0, drop_level=True).copy()
        else:
            return pd.DataFrame()
    else:
        out = downloaded.copy()

    out = out.dropna(how="all")
    return out


def classify_signal(df: pd.DataFrame, short_window: int, medium_window: int) -> dict | None:
    if "Close" not in df.columns:
        return None

    work = df.copy()
    work["SMA_short"] = work["Close"].rolling(short_window).mean()
    work["SMA_medium"] = work["Close"].rolling(medium_window).mean()

    # Extra context that can later be used by the AI review.
    work["SMA_200"] = work["Close"].rolling(200).mean()
    work["Vol_20"] = work["Volume"].rolling(20).mean() if "Volume" in work.columns else math.nan

    usable = work.dropna(subset=["SMA_short", "SMA_medium"])
    if len(usable) < 2:
        return None

    prev = usable.iloc[-2]
    curr = usable.iloc[-1]

    buy = prev["SMA_short"] <= prev["SMA_medium"] and curr["SMA_short"] > curr["SMA_medium"]
    sell = prev["SMA_short"] >= prev["SMA_medium"] and curr["SMA_short"] < curr["SMA_medium"]

    signal = "BUY" if buy else "SELL" if sell else "NONE"

    sma200 = curr.get("SMA_200", math.nan)
    vol20 = curr.get("Vol_20", math.nan)
    volume = curr.get("Volume", math.nan)

    return {
        "date": usable.index[-1].date().isoformat(),
        "signal": signal,
        "close": float(curr["Close"]),
        "short_sma": float(curr["SMA_short"]),
        "medium_sma": float(curr["SMA_medium"]),
        "sma_200": None if pd.isna(sma200) else float(sma200),
        "above_200_sma": None if pd.isna(sma200) else bool(curr["Close"] > sma200),
        "volume": None if pd.isna(volume) else float(volume),
        "volume_20d_avg": None if pd.isna(vol20) else float(vol20),
        "volume_ratio": None if pd.isna(volume) or pd.isna(vol20) or vol20 == 0 else float(volume / vol20),
        "cross_gap_pct": float((curr["SMA_short"] / curr["SMA_medium"] - 1.0) * 100.0),
    }


def chunks(seq: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def download_and_scan(universe: pd.DataFrame, short_window: int, medium_window: int,
                      period: str, batch_size: int) -> pd.DataFrame:
    company_by_ticker = universe.set_index("ticker")["company"].to_dict()
    market_by_ticker = universe.set_index("ticker")["market"].to_dict()
    tickers = universe["ticker"].tolist()
    rows = []

    for batch in chunks(tickers, batch_size):
        data = yf.download(
            tickers=batch,
            period=period,
            interval="1d",
            auto_adjust=True,
            group_by="column",
            threads=True,
            progress=False,
        )

        for ticker in batch:
            one = extract_one_ticker(data, ticker)
            result = classify_signal(one, short_window, medium_window)
            if result is None:
                continue
            result.update({
                "ticker": ticker,
                "company": company_by_ticker[ticker],
                "market": market_by_ticker[ticker],
            })
            rows.append(result)

    if not rows:
        return pd.DataFrame()

    cols = [
        "date", "market", "ticker", "company", "signal", "close",
        "short_sma", "medium_sma", "cross_gap_pct", "sma_200",
        "above_200_sma", "volume", "volume_20d_avg", "volume_ratio",
    ]
    return pd.DataFrame(rows)[cols].sort_values(["signal", "market", "ticker"])


PROMPT_HEADER = """\
# BUY-SIGNAL FUNDAMENTAL + NEWS REVIEW

You are reviewing stocks that have ALREADY triggered a deterministic technical BUY signal.
Do not change the technical signal and do not invent missing data.

## Technical rule used by the screener
BUY = the short daily simple moving average crossed from at-or-below the medium daily
simple moving average on the previous completed candle to strictly above it on the
latest completed daily candle.

The AI review is NOT a forecast of investment return. Its percentage is an
EVIDENCE-SUPPORT SCORE indicating how strongly current fundamentals, market context,
and recent material news support the technical BUY signal.

## Research requirements
For EACH stock below:
1. Identify the company and confirm the ticker/exchange.
2. Use the latest available reported financial statements and company filings.
3. Review material company-specific news published in the last 30 calendar days.
4. Prefer primary sources: company investor-relations releases, regulatory filings,
   exchange announcements, SEC filings, and official results presentations.
5. Use high-quality financial/news sources only as secondary confirmation.
6. State the reporting period/date for every financial figure used.
7. Do not treat analyst opinion, social-media sentiment, rumours or unsourced claims
   as company fundamentals.
8. If a required datum cannot be established reliably, mark it UNKNOWN and exclude
   that item from the denominator rather than guessing.

## Deterministic scoring rubric
Score each applicable item exactly as specified.

A. Revenue trajectory — maximum 15
- Latest comparable revenue growth > +10% YoY: 15
- > 0% to +10%: 10
- 0% to -10%: 5
- < -10%: 0

B. Profitability trajectory — maximum 15
Use reported operating profit or net income, choosing the measure most consistently
reported by the company. Compare like-for-like periods.
- Profitable AND profit growth > +10% YoY: 15
- Profitable AND growth between 0% and +10%: 10
- Profitable BUT profit declined: 5
- Loss-making: 0

C. Free-cash-flow quality — maximum 15
- Positive FCF and improved YoY: 15
- Positive FCF but declined YoY: 10
- Approximately break-even: 5
- Negative FCF: 0

D. Balance-sheet / financing risk — maximum 15
For non-financial companies use net-debt/EBITDA where reliably available:
- < 1.5x: 15
- 1.5x to < 2.5x: 10
- 2.5x to < 3.5x: 5
- >= 3.5x OR explicit near-term financing distress: 0
For banks/insurers or where this metric is not meaningful, mark UNKNOWN and explain.

E. Guidance / outlook — maximum 15
- Management explicitly raised current-year guidance: 15
- Reaffirmed / maintained guidance: 10
- No formal guidance or genuinely neutral update: 7
- Lowered guidance: 0

F. Material news, last 30 days — maximum 15
Classify ONLY company-specific, decision-relevant developments:
- Net materially positive: 15
- Mixed / no material development: 8
- Net materially negative: 0

G. Technical context supplied by screener — maximum 10
- Close above 200-day SMA: +5
- Latest daily volume >= 1.20 x trailing 20-day average volume: +5
If either datum is unavailable, exclude that sub-item from the denominator.

## Final score
EVIDENCE_SUPPORT_PCT = 100 * points_awarded / points_available
Round to nearest integer.

Then map:
- 80–100: STRONGLY SUPPORTS BUY SIGNAL
- 65–79: SUPPORTS BUY SIGNAL
- 50–64: NEUTRAL / MIXED
- 35–49: WEAKENS BUY SIGNAL
- 0–34: STRONGLY WEAKENS BUY SIGNAL

## Required output
Return:
1. A concise markdown table, one row per stock, sorted by EVIDENCE_SUPPORT_PCT descending.
2. Then a JSON array using the exact schema below.

Table columns:
Ticker | Company | Market | Signal date | Evidence support % | Classification |
Revenue | Profitability | FCF | Balance sheet | Guidance | 30d news | Key reason

Exact JSON schema:
[
  {
    "ticker": "string",
    "company": "string",
    "market": "US or UK",
    "signal_date": "YYYY-MM-DD",
    "evidence_support_pct": 0,
    "classification": "string",
    "points_awarded": 0,
    "points_available": 0,
    "scores": {
      "revenue": {"score": 0, "max": 15, "status": "KNOWN|UNKNOWN", "evidence": "string"},
      "profitability": {"score": 0, "max": 15, "status": "KNOWN|UNKNOWN", "evidence": "string"},
      "free_cash_flow": {"score": 0, "max": 15, "status": "KNOWN|UNKNOWN", "evidence": "string"},
      "balance_sheet": {"score": 0, "max": 15, "status": "KNOWN|UNKNOWN", "evidence": "string"},
      "guidance": {"score": 0, "max": 15, "status": "KNOWN|UNKNOWN", "evidence": "string"},
      "recent_news": {"score": 0, "max": 15, "status": "KNOWN|UNKNOWN", "evidence": "string"},
      "above_200_sma": {"score": 0, "max": 5, "status": "KNOWN|UNKNOWN", "evidence": "string"},
      "volume_confirmation": {"score": 0, "max": 5, "status": "KNOWN|UNKNOWN", "evidence": "string"}
    },
    "key_positive": "string",
    "key_risk": "string",
    "sources": [
      {"title": "string", "publisher": "string", "date": "YYYY-MM-DD", "url": "string"}
    ]
  }
]

Do not add points for anything outside the rubric.
Do not reinterpret missing information optimistically or pessimistically.
Do not call the score a probability that the share price will rise.
"""


def render_stock_block(row: pd.Series) -> str:
    return f"""
### {row['ticker']} — {row['company']} ({row['market']})
- Signal date: {row['date']}
- Close: {row['close']:.4f}
- Short SMA: {row['short_sma']:.4f}
- Medium SMA: {row['medium_sma']:.4f}
- Short-vs-medium gap: {row['cross_gap_pct']:.4f}%
- 200-day SMA: {row['sma_200'] if pd.notna(row['sma_200']) else 'UNKNOWN'}
- Close above 200-day SMA: {row['above_200_sma'] if pd.notna(row['above_200_sma']) else 'UNKNOWN'}
- Latest volume / 20-day average volume: {row['volume_ratio'] if pd.notna(row['volume_ratio']) else 'UNKNOWN'}
"""


def write_ai_prompt(buys: pd.DataFrame, path: Path) -> None:
    text = PROMPT_HEADER
    text += "\n\n# STOCKS TO REVIEW\n"
    if buys.empty:
        text += "\nNo BUY signals were generated in this run.\n"
    else:
        for _, row in buys.iterrows():
            text += render_stock_block(row)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--short", type=int, default=20, help="Short SMA window; default 20")
    parser.add_argument("--medium", type=int, default=50, help="Medium SMA window; default 50")
    parser.add_argument("--period", default="2y", help="History requested from yfinance; default 2y")
    parser.add_argument("--batch-size", type=int, default=75)
    parser.add_argument("--no-sp500", action="store_true")
    parser.add_argument("--no-ftse100", action="store_true")
    parser.add_argument("--custom-csv", help="Optional CSV with ticker,company,market")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    if args.short <= 0 or args.medium <= 0:
        raise ValueError("--short and --medium must be positive")
    if args.short >= args.medium:
        raise ValueError("--short must be smaller than --medium")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    universe = build_universe(
        include_sp500=not args.no_sp500,
        include_ftse100=not args.no_ftse100,
        custom_csv=args.custom_csv,
    )

    print(f"Universe: {len(universe)} stocks")
    print(f"Downloading daily candles and testing SMA({args.short}) / SMA({args.medium}) crosses...")

    results = download_and_scan(
        universe=universe,
        short_window=args.short,
        medium_window=args.medium,
        period=args.period,
        batch_size=args.batch_size,
    )

    if results.empty:
        print("No usable results.")
        return

    buys = results[results["signal"] == "BUY"].copy()
    sells = results[results["signal"] == "SELL"].copy()

    results.to_csv(output / "all_signals.csv", index=False)
    buys.to_csv(output / "buy_signals.csv", index=False)
    sells.to_csv(output / "sell_signals.csv", index=False)
    write_ai_prompt(buys, output / "buy_signal_review_prompt.txt")

    display_cols = ["market", "ticker", "company", "date", "close", "short_sma", "medium_sma", "volume_ratio"]
    print("\nNEW BUY SIGNALS")
    if buys.empty:
        print("None.")
    else:
        print(buys[display_cols].to_string(index=False))

    print("\nNEW SELL SIGNALS")
    if sells.empty:
        print("None.")
    else:
        print(sells[display_cols].to_string(index=False))

    print(f"\nWritten to: {output.resolve()}")
    print("Upload buy_signal_review_prompt.txt to ChatGPT/Codex for the qualitative review.")


if __name__ == "__main__":
    main()
