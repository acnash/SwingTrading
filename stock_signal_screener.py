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
  output/actionable_buy_signals.csv
  output/wait_for_pullback_signals.csv
  output/wait_for_confirmation_signals.csv
  output/wait_for_volume_confirmation_signals.csv
  output/sell_signals.csv
  output/failed_symbols.csv
  output/buy_signal_review_prompt.txt

This is a research/screening tool, not investment advice.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import time
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import yfinance as yf


WIKIPEDIA_USER_AGENT = "SwingTrading/1.0 constituent-table reader"


def read_html_tables(url: str) -> list[pd.DataFrame]:
    response = requests.get(
        url,
        headers={"User-Agent": WIKIPEDIA_USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    return pd.read_html(StringIO(response.text))


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
    tables = read_html_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = tables[0][["Symbol", "Security"]].copy()
    df.columns = ["ticker", "company"]
    df["ticker"] = df["ticker"].map(normalise_us_symbol)
    df["market"] = "US"
    return df


def get_ftse100() -> pd.DataFrame:
    tables = read_html_tables("https://en.wikipedia.org/wiki/FTSE_100_Index")
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


def add_vfi_columns(
    frame: pd.DataFrame,
    period: int = 130,
    volatility_period: int = 30,
    volume_cap: float = 2.5,
    coefficient: float = 0.2,
    signal_period: int = 5,
) -> pd.DataFrame:
    """Add the conventional Volume Flow Indicator and its EMA signal line."""
    work = frame.copy()
    required = {"High", "Low", "Close", "Volume"}
    if not required.issubset(work.columns):
        work["VFI"] = math.nan
        work["VFI_signal"] = math.nan
        return work

    typical = (work["High"] + work["Low"] + work["Close"]) / 3.0
    log_change = pd.Series(math.nan, index=work.index, dtype=float)
    positive = (typical > 0) & (typical.shift(1) > 0)
    log_change.loc[positive] = (
        typical.loc[positive].map(math.log)
        - typical.shift(1).loc[positive].map(math.log)
    )
    volatility = log_change.rolling(volatility_period).std()
    cutoff = coefficient * volatility * work["Close"]
    price_change = typical.diff()
    average_volume = work["Volume"].rolling(period).mean().shift(1)
    capped_volume = pd.concat(
        [work["Volume"], average_volume * volume_cap], axis=1
    ).min(axis=1)
    signed_volume = pd.Series(0.0, index=work.index)
    signed_volume = signed_volume.mask(price_change > cutoff, capped_volume)
    signed_volume = signed_volume.mask(price_change < -cutoff, -capped_volume)
    work["VFI"] = signed_volume.rolling(period).sum() / average_volume
    work["VFI_signal"] = work["VFI"].ewm(
        span=signal_period, adjust=False, min_periods=signal_period
    ).mean()
    return work


def vfi_confirmation_score(work: pd.DataFrame) -> tuple[int | None, str]:
    """Return a deterministic 0-100 VFI confirmation score and classification."""
    usable = work.dropna(subset=["VFI", "VFI_signal"])
    if len(usable) < 21:
        return None, "UNAVAILABLE"

    vfi = usable["VFI"]
    signal = usable["VFI_signal"]
    score = 0
    score += 25 if vfi.iloc[-1] > 0 else 0
    score += 20 if vfi.iloc[-1] > signal.iloc[-1] else 0
    recent_cross = (
        (vfi.shift(1) <= signal.shift(1)) & (vfi > signal)
    ).tail(3).any()
    score += 20 if recent_cross else 0
    score += 15 if vfi.iloc[-1] > vfi.iloc[-6] else 0
    score += 10 if vfi.iloc[-1] > vfi.iloc[-21] else 0
    twenty_day_high = vfi.tail(20).max()
    near_high = vfi.iloc[-1] >= twenty_day_high - 0.1 * abs(twenty_day_high)
    score += 10 if near_high else 0

    if score >= 80:
        classification = "STRONG_ACCUMULATION"
    elif score >= 65:
        classification = "SUPPORTS_BUY"
    elif score >= 50:
        classification = "MIXED_EARLY_ACCUMULATION"
    elif score >= 35:
        classification = "WEAK_VOLUME_SUPPORT"
    else:
        classification = "DISTRIBUTION_OR_ABSENT_SUPPORT"
    return score, classification


def classify_signal(
    df: pd.DataFrame,
    short_window: int,
    medium_window: int,
    rsi_period: int = 14,
    max_rsi: float = 68.0,
    max_short_extension_pct: float = 4.0,
    max_five_day_gain_pct: float = 8.0,
    resistance_lookback: int = 60,
    resistance_proximity_pct: float = 2.0,
    high_low_lookback: int = 252,
    max_52_week_high_distance_pct: float = 1.0,
    volume_confirmation_ratio: float = 1.2,
    volume_confirmation_days: int = 2,
    volume_confirmation_window: int = 3,
    vfi_period: int = 130,
    vfi_volatility_period: int = 30,
    vfi_volume_cap: float = 2.5,
    vfi_coefficient: float = 0.2,
    vfi_signal_period: int = 5,
    min_vfi_buy_score: int = 50,
) -> dict | None:
    if "Close" not in df.columns:
        return None

    work = df.copy()
    work["SMA_short"] = work["Close"].rolling(short_window).mean()
    work["SMA_medium"] = work["Close"].rolling(medium_window).mean()

    # Extra context that can later be used by the AI review.
    work["SMA_200"] = work["Close"].rolling(200).mean()
    work["Vol_20"] = work["Volume"].rolling(20).mean() if "Volume" in work.columns else math.nan
    delta = work["Close"].diff()
    average_gain = delta.clip(lower=0).ewm(alpha=1 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    average_loss = -delta.clip(upper=0).ewm(alpha=1 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    relative_strength = average_gain / average_loss
    work["RSI"] = 100 - (100 / (1 + relative_strength))
    work["Five_day_gain_pct"] = work["Close"].pct_change(5) * 100
    work["Prior_resistance"] = work["Close"].shift(1).rolling(resistance_lookback).max()
    high_prices = work["High"] if "High" in work.columns else work["Close"]
    low_prices = work["Low"] if "Low" in work.columns else work["Close"]
    work["High_52w"] = high_prices.rolling(high_low_lookback).max()
    work["Low_52w"] = low_prices.rolling(high_low_lookback).min()
    if "Volume" in work.columns:
        work["Volume_ratio"] = work["Volume"] / work["Vol_20"]
    work = add_vfi_columns(
        work, period=vfi_period, volatility_period=vfi_volatility_period,
        volume_cap=vfi_volume_cap, coefficient=vfi_coefficient,
        signal_period=vfi_signal_period,
    )

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
    rsi = curr.get("RSI", math.nan)
    five_day_gain_pct = curr.get("Five_day_gain_pct", math.nan)
    prior_resistance = curr.get("Prior_resistance", math.nan)
    high_52w = curr.get("High_52w", math.nan)
    low_52w = curr.get("Low_52w", math.nan)
    vfi = curr.get("VFI", math.nan)
    vfi_signal = curr.get("VFI_signal", math.nan)
    vfi_score, vfi_classification = vfi_confirmation_score(usable)
    short_extension_pct = (curr["Close"] / curr["SMA_short"] - 1.0) * 100.0
    resistance_distance_pct = (
        (prior_resistance - curr["Close"]) / curr["Close"] * 100.0
        if not pd.isna(prior_resistance) and curr["Close"] != 0 else math.nan
    )
    distance_from_52_week_high_pct = (
        (high_52w - curr["Close"]) / high_52w * 100.0
        if not pd.isna(high_52w) and high_52w != 0 else math.nan
    )
    position_in_52_week_range_pct = (
        (curr["Close"] - low_52w) / (high_52w - low_52w) * 100.0
        if (
            not pd.isna(high_52w)
            and not pd.isna(low_52w)
            and high_52w != low_52w
        )
        else math.nan
    )
    at_52_week_peak = (
        not pd.isna(distance_from_52_week_high_pct)
        and distance_from_52_week_high_pct <= max_52_week_high_distance_pct
    )

    recent_volume_ratios = (
        usable["Volume_ratio"].tail(volume_confirmation_window)
        if "Volume_ratio" in usable.columns else pd.Series(dtype=float)
    )
    confirmed_volume_days = int((recent_volume_ratios >= volume_confirmation_ratio).sum())
    persistent_volume_confirmation = (
        confirmed_volume_days >= volume_confirmation_days
        if len(recent_volume_ratios.dropna()) >= volume_confirmation_days else None
    )

    overextension_reasons = []
    if not pd.isna(rsi) and rsi >= max_rsi:
        overextension_reasons.append(f"RSI {rsi:.1f} >= {max_rsi:.1f}")
    if short_extension_pct >= max_short_extension_pct:
        overextension_reasons.append(
            f"close {short_extension_pct:.1f}% above short SMA >= {max_short_extension_pct:.1f}%"
        )
    if not pd.isna(five_day_gain_pct) and five_day_gain_pct >= max_five_day_gain_pct:
        overextension_reasons.append(
            f"five-day gain {five_day_gain_pct:.1f}% >= {max_five_day_gain_pct:.1f}%"
        )
    if at_52_week_peak:
        overextension_reasons.append(
            "close "
            f"{distance_from_52_week_high_pct:.1f}% below 52-week high "
            f"<= {max_52_week_high_distance_pct:.1f}%"
        )

    near_resistance = (
        not pd.isna(resistance_distance_pct)
        and abs(resistance_distance_pct) <= resistance_proximity_pct
    )
    if buy and overextension_reasons:
        entry_status = "WAIT_FOR_PULLBACK"
    elif buy and near_resistance and persistent_volume_confirmation is not True:
        entry_status = "WAIT_FOR_CONFIRMATION"
    elif buy and vfi_score is not None and vfi_score < min_vfi_buy_score:
        entry_status = "WAIT_FOR_VOLUME_CONFIRMATION"
    elif buy:
        entry_status = "ACTIONABLE_BUY"
    else:
        entry_status = "NOT_APPLICABLE"

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
        "rsi_14": None if pd.isna(rsi) else float(rsi),
        "short_sma_extension_pct": float(short_extension_pct),
        "five_day_gain_pct": None if pd.isna(five_day_gain_pct) else float(five_day_gain_pct),
        "prior_resistance": None if pd.isna(prior_resistance) else float(prior_resistance),
        "resistance_distance_pct": None if pd.isna(resistance_distance_pct) else float(resistance_distance_pct),
        "near_resistance": bool(near_resistance),
        "fifty_two_week_high": None if pd.isna(high_52w) else float(high_52w),
        "fifty_two_week_low": None if pd.isna(low_52w) else float(low_52w),
        "distance_from_52_week_high_pct": (
            None
            if pd.isna(distance_from_52_week_high_pct)
            else float(distance_from_52_week_high_pct)
        ),
        "position_in_52_week_range_pct": (
            None
            if pd.isna(position_in_52_week_range_pct)
            else float(position_in_52_week_range_pct)
        ),
        "at_52_week_peak": bool(at_52_week_peak),
        "confirmed_volume_days": confirmed_volume_days,
        "persistent_volume_confirmation": persistent_volume_confirmation,
        "vfi": None if pd.isna(vfi) else float(vfi),
        "vfi_signal": None if pd.isna(vfi_signal) else float(vfi_signal),
        "vfi_buy_index": vfi_score,
        "vfi_classification": vfi_classification,
        "entry_status": entry_status,
        "entry_warning": "; ".join(overextension_reasons),
    }


def chunks(seq: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def ticker_cache_path(cache_dir: Path, ticker: str, period: str) -> Path:
    safe_ticker = re.sub(r"[^A-Za-z0-9._-]+", "_", ticker)
    safe_period = re.sub(r"[^A-Za-z0-9._-]+", "_", period)
    return cache_dir / safe_period / f"{safe_ticker}.csv"


def load_cached_ticker(cache_dir: Path, ticker: str, period: str,
                       max_age_hours: float) -> pd.DataFrame:
    path = ticker_cache_path(cache_dir, ticker, period)
    if not path.exists() or max_age_hours <= 0:
        return pd.DataFrame()
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    if age_hours > max_age_hours:
        return pd.DataFrame()
    try:
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()
    return cached if "Close" in cached.columns else pd.DataFrame()


def save_cached_ticker(cache_dir: Path, ticker: str, period: str,
                       data: pd.DataFrame) -> None:
    if data.empty:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = ticker_cache_path(cache_dir, ticker, period)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path)


def fetch_tickers(tickers: list[str], period: str, threads: int) -> pd.DataFrame:
    return yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="column",
        threads=threads,
        progress=False,
        timeout=30,
    )


def retry_failed_ticker(ticker: str, period: str, threads: int, max_retries: int,
                        retry_base_delay: float) -> tuple[pd.DataFrame, int, str]:
    last_error = "No usable price history returned"
    for retry_number in range(1, max_retries + 1):
        delay = min(retry_base_delay * (2 ** (retry_number - 1)), 60.0)
        delay += random.uniform(0, min(1.0, delay * 0.25)) if delay > 0 else 0
        if delay > 0:
            print(f"  Retrying {ticker} in {delay:.1f}s ({retry_number}/{max_retries})...")
            time.sleep(delay)
        try:
            downloaded = fetch_tickers([ticker], period, threads)
            one = extract_one_ticker(downloaded, ticker)
            if not one.empty:
                return one, retry_number, ""
        except Exception as exc:  # yfinance raises several transient transport errors.
            last_error = f"{type(exc).__name__}: {exc}"
    return pd.DataFrame(), max_retries, last_error


def download_and_scan(universe: pd.DataFrame, short_window: int, medium_window: int,
                      period: str, batch_size: int, threads: int = 4,
                      min_pause: float = 2.0, max_pause: float = 4.0,
                      max_retries: int = 3, retry_base_delay: float = 2.0,
                      cache_dir: Path = Path(".cache/market_data"),
                      cache_max_age_hours: float = 12.0,
                      refresh_cache: bool = False,
                      signal_options: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    company_by_ticker = universe.set_index("ticker")["company"].to_dict()
    market_by_ticker = universe.set_index("ticker")["market"].to_dict()
    tickers = universe["ticker"].tolist()
    rows = []
    failures = []
    ticker_batches = list(chunks(tickers, batch_size))

    for batch_number, batch in enumerate(ticker_batches, start=1):
        print(f"Batch {batch_number}/{len(ticker_batches)}: {len(batch)} symbols")
        ticker_data: dict[str, pd.DataFrame] = {}
        live_tickers = []
        for ticker in batch:
            cached = pd.DataFrame() if refresh_cache else load_cached_ticker(
                cache_dir, ticker, period, cache_max_age_hours
            )
            if cached.empty:
                live_tickers.append(ticker)
            else:
                ticker_data[ticker] = cached

        downloaded = pd.DataFrame()
        batch_error = ""
        if live_tickers:
            try:
                downloaded = fetch_tickers(live_tickers, period, threads)
            except Exception as exc:
                batch_error = f"{type(exc).__name__}: {exc}"

            for ticker in live_tickers:
                one = extract_one_ticker(downloaded, ticker)
                if one.empty:
                    one, retry_count, retry_error = retry_failed_ticker(
                        ticker=ticker,
                        period=period,
                        threads=threads,
                        max_retries=max_retries,
                        retry_base_delay=retry_base_delay,
                    )
                    if one.empty:
                        failures.append({
                            "ticker": ticker,
                            "company": company_by_ticker[ticker],
                            "market": market_by_ticker[ticker],
                            "reason": retry_error or batch_error or "No usable price history returned",
                            "retry_attempts": retry_count,
                        })
                        continue
                ticker_data[ticker] = one
                save_cached_ticker(cache_dir, ticker, period, one)

        for ticker in batch:
            one = ticker_data.get(ticker, pd.DataFrame())
            if one.empty:
                continue
            result = classify_signal(one, short_window, medium_window, **(signal_options or {}))
            if result is None:
                failures.append({
                    "ticker": ticker,
                    "company": company_by_ticker[ticker],
                    "market": market_by_ticker[ticker],
                    "reason": "Insufficient usable history for moving averages",
                    "retry_attempts": 0,
                })
                continue
            result.update({
                "ticker": ticker,
                "company": company_by_ticker[ticker],
                "market": market_by_ticker[ticker],
            })
            rows.append(result)

        if live_tickers and batch_number < len(ticker_batches) and max_pause > 0:
            pause = random.uniform(min_pause, max_pause)
            print(f"  Pausing {pause:.1f}s before the next batch...")
            time.sleep(pause)

    cols = [
        "date", "market", "ticker", "company", "signal", "close",
        "short_sma", "medium_sma", "cross_gap_pct", "sma_200",
        "above_200_sma", "volume", "volume_20d_avg", "volume_ratio",
        "rsi_14", "short_sma_extension_pct", "five_day_gain_pct",
        "prior_resistance", "resistance_distance_pct", "near_resistance",
        "fifty_two_week_high", "fifty_two_week_low",
        "distance_from_52_week_high_pct", "position_in_52_week_range_pct",
        "at_52_week_peak",
        "confirmed_volume_days", "persistent_volume_confirmation", "vfi",
        "vfi_signal", "vfi_buy_index", "vfi_classification",
        "entry_status", "entry_warning",
    ]
    results = (
        pd.DataFrame(rows)[cols].sort_values(["signal", "market", "ticker"])
        if rows else pd.DataFrame(columns=cols)
    )
    failure_cols = ["ticker", "company", "market", "reason", "retry_attempts"]
    failed = (
        pd.DataFrame(failures)[failure_cols].drop_duplicates(["market", "ticker"], keep="last")
        if failures else pd.DataFrame(columns=failure_cols)
    )
    return results, failed


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
    vfi = row.get("vfi", math.nan)
    vfi_signal = row.get("vfi_signal", math.nan)
    vfi_buy_index = row.get("vfi_buy_index", math.nan)
    vfi_classification = row.get("vfi_classification", "UNAVAILABLE")
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
- RSI(14): {row['rsi_14'] if pd.notna(row['rsi_14']) else 'UNKNOWN'}
- Close distance above short SMA: {row['short_sma_extension_pct']:.2f}%
- Five-day price change: {row['five_day_gain_pct'] if pd.notna(row['five_day_gain_pct']) else 'UNKNOWN'}%
- Prior 60-day resistance: {row['prior_resistance'] if pd.notna(row['prior_resistance']) else 'UNKNOWN'}
- Distance to prior resistance: {row['resistance_distance_pct'] if pd.notna(row['resistance_distance_pct']) else 'UNKNOWN'}%
- 52-week high: {row['fifty_two_week_high'] if pd.notna(row['fifty_two_week_high']) else 'UNKNOWN'}
- 52-week low: {row['fifty_two_week_low'] if pd.notna(row['fifty_two_week_low']) else 'UNKNOWN'}
- Distance below 52-week high: {row['distance_from_52_week_high_pct'] if pd.notna(row['distance_from_52_week_high_pct']) else 'UNKNOWN'}%
- Position in 52-week range: {row['position_in_52_week_range_pct'] if pd.notna(row['position_in_52_week_range_pct']) else 'UNKNOWN'}%
- At 52-week peak: {row['at_52_week_peak']}
- Persistent volume confirmation: {row['persistent_volume_confirmation']}
- VFI: {vfi if pd.notna(vfi) else 'UNKNOWN'}
- VFI signal line: {vfi_signal if pd.notna(vfi_signal) else 'UNKNOWN'}
- VFI BUY confirmation index: {vfi_buy_index if pd.notna(vfi_buy_index) else 'UNKNOWN'}
- VFI classification: {vfi_classification}
- Entry status: {row['entry_status']}
- Entry warning: {row['entry_warning'] or 'None'}
"""


def write_ai_prompt(buys: pd.DataFrame, path: Path, reviewer_number: int | None = None) -> None:
    text = PROMPT_HEADER
    if reviewer_number is not None:
        text += f"""

# INDEPENDENT REVIEW {reviewer_number} OF 3

Complete this review independently. Do not inspect, copy, reconcile or average the
other reviewers' answers. Return your own rubric scores and supporting evidence.
"""
    text += "\n\n# STOCKS TO REVIEW\n"
    if buys.empty:
        text += "\nNo BUY signals were generated in this run.\n"
    else:
        for _, row in buys.iterrows():
            text += render_stock_block(row)
    path.write_text(text, encoding="utf-8")


def write_three_ai_prompts(buys: pd.DataFrame, output: Path) -> None:
    """Write three isolated prompts for the required independent reviews."""
    for reviewer_number in range(1, 4):
        write_ai_prompt(
            buys,
            output / f"buy_signal_review_prompt_{reviewer_number}.txt",
            reviewer_number,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--short", type=int, default=20, help="Short SMA window; default 20")
    parser.add_argument("--medium", type=int, default=50, help="Medium SMA window; default 50")
    parser.add_argument("--period", default="2y", help="History requested from yfinance; default 2y")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--threads", type=int, default=4, help="Maximum yfinance workers; default 4")
    parser.add_argument("--min-pause", type=float, default=2.0, help="Minimum seconds between live batches")
    parser.add_argument("--max-pause", type=float, default=4.0, help="Maximum seconds between live batches")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries for failed symbols only")
    parser.add_argument("--retry-base-delay", type=float, default=2.0, help="Initial exponential-backoff delay")
    parser.add_argument("--cache-dir", default=".cache/market_data")
    parser.add_argument("--cache-max-age-hours", type=float, default=12.0)
    parser.add_argument("--refresh-cache", action="store_true", help="Ignore cached price histories")
    parser.add_argument("--no-sp500", action="store_true")
    parser.add_argument("--no-ftse100", action="store_true")
    parser.add_argument("--custom-csv", help="Optional CSV with ticker,company,market")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--max-rsi", type=float, default=68.0)
    parser.add_argument("--max-short-extension-pct", type=float, default=4.0)
    parser.add_argument("--max-five-day-gain-pct", type=float, default=8.0)
    parser.add_argument("--resistance-lookback", type=int, default=60)
    parser.add_argument("--resistance-proximity-pct", type=float, default=2.0)
    parser.add_argument("--high-low-lookback", type=int, default=252)
    parser.add_argument("--max-52-week-high-distance-pct", type=float, default=1.0)
    parser.add_argument("--volume-confirmation-ratio", type=float, default=1.2)
    parser.add_argument("--volume-confirmation-days", type=int, default=2)
    parser.add_argument("--volume-confirmation-window", type=int, default=3)
    parser.add_argument("--vfi-period", type=int, default=130)
    parser.add_argument("--vfi-volatility-period", type=int, default=30)
    parser.add_argument("--vfi-volume-cap", type=float, default=2.5)
    parser.add_argument("--vfi-coefficient", type=float, default=0.2)
    parser.add_argument("--vfi-signal-period", type=int, default=5)
    parser.add_argument("--min-vfi-buy-score", type=int, default=50)
    args = parser.parse_args()

    if args.short <= 0 or args.medium <= 0:
        raise ValueError("--short and --medium must be positive")
    if args.short >= args.medium:
        raise ValueError("--short must be smaller than --medium")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if args.min_pause < 0 or args.max_pause < args.min_pause:
        raise ValueError("pause values must satisfy 0 <= --min-pause <= --max-pause")
    if args.max_retries < 0 or args.retry_base_delay < 0:
        raise ValueError("retry values must be non-negative")
    if args.cache_max_age_hours < 0:
        raise ValueError("--cache-max-age-hours must be non-negative")
    if min(args.resistance_lookback, args.high_low_lookback, args.volume_confirmation_window) <= 0:
        raise ValueError("lookback and confirmation window values must be positive")
    if args.max_52_week_high_distance_pct < 0:
        raise ValueError("--max-52-week-high-distance-pct must be non-negative")
    if not 1 <= args.volume_confirmation_days <= args.volume_confirmation_window:
        raise ValueError("--volume-confirmation-days must be within the confirmation window")
    if min(args.vfi_period, args.vfi_volatility_period, args.vfi_signal_period) <= 0:
        raise ValueError("VFI periods must be positive")
    if args.vfi_volume_cap <= 0 or args.vfi_coefficient <= 0:
        raise ValueError("VFI volume cap and coefficient must be positive")
    if not 0 <= args.min_vfi_buy_score <= 100:
        raise ValueError("--min-vfi-buy-score must be between 0 and 100")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    universe = build_universe(
        include_sp500=not args.no_sp500,
        include_ftse100=not args.no_ftse100,
        custom_csv=args.custom_csv,
    )

    print(f"Universe: {len(universe)} stocks")
    print(f"Downloading daily candles and testing SMA({args.short}) / SMA({args.medium}) crosses...")

    results, failures = download_and_scan(
        universe=universe,
        short_window=args.short,
        medium_window=args.medium,
        period=args.period,
        batch_size=args.batch_size,
        threads=args.threads,
        min_pause=args.min_pause,
        max_pause=args.max_pause,
        max_retries=args.max_retries,
        retry_base_delay=args.retry_base_delay,
        cache_dir=Path(args.cache_dir),
        cache_max_age_hours=args.cache_max_age_hours,
        refresh_cache=args.refresh_cache,
        signal_options={
            "max_rsi": args.max_rsi,
            "max_short_extension_pct": args.max_short_extension_pct,
            "max_five_day_gain_pct": args.max_five_day_gain_pct,
            "resistance_lookback": args.resistance_lookback,
            "resistance_proximity_pct": args.resistance_proximity_pct,
            "high_low_lookback": args.high_low_lookback,
            "max_52_week_high_distance_pct": args.max_52_week_high_distance_pct,
            "volume_confirmation_ratio": args.volume_confirmation_ratio,
            "volume_confirmation_days": args.volume_confirmation_days,
            "volume_confirmation_window": args.volume_confirmation_window,
            "vfi_period": args.vfi_period,
            "vfi_volatility_period": args.vfi_volatility_period,
            "vfi_volume_cap": args.vfi_volume_cap,
            "vfi_coefficient": args.vfi_coefficient,
            "vfi_signal_period": args.vfi_signal_period,
            "min_vfi_buy_score": args.min_vfi_buy_score,
        },
    )

    failures.to_csv(output / "failed_symbols.csv", index=False)

    if results.empty:
        print("No usable results.")
        print(f"Failure audit written to: {(output / 'failed_symbols.csv').resolve()}")
        return

    buys = results[results["signal"] == "BUY"].copy()
    sells = results[results["signal"] == "SELL"].copy()
    actionable_buys = buys[buys["entry_status"] == "ACTIONABLE_BUY"].copy()
    pullback_buys = buys[buys["entry_status"] == "WAIT_FOR_PULLBACK"].copy()
    confirmation_buys = buys[buys["entry_status"] == "WAIT_FOR_CONFIRMATION"].copy()
    volume_wait_buys = buys[buys["entry_status"] == "WAIT_FOR_VOLUME_CONFIRMATION"].copy()

    results.to_csv(output / "all_signals.csv", index=False)
    buys.to_csv(output / "buy_signals.csv", index=False)
    actionable_buys.to_csv(output / "actionable_buy_signals.csv", index=False)
    pullback_buys.to_csv(output / "wait_for_pullback_signals.csv", index=False)
    confirmation_buys.to_csv(output / "wait_for_confirmation_signals.csv", index=False)
    volume_wait_buys.to_csv(output / "wait_for_volume_confirmation_signals.csv", index=False)
    sells.to_csv(output / "sell_signals.csv", index=False)
    write_three_ai_prompts(buys, output)

    display_cols = ["market", "ticker", "company", "date", "close", "short_sma", "medium_sma", "rsi_14", "volume_ratio", "vfi_buy_index", "vfi_classification", "entry_status"]
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
    print("Run the three numbered review prompts in separate Codex tasks, then aggregate their JSON files.")


if __name__ == "__main__":
    main()
