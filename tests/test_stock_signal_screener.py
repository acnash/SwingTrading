from pathlib import Path

import pandas as pd
import stock_signal_screener as screener

from stock_signal_screener import (
    classify_signal,
    download_and_scan,
    load_custom_csv,
    load_cached_ticker,
    normalise_uk_symbol,
    normalise_us_symbol,
    save_cached_ticker,
)


def price_frame(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {"Close": closes, "Volume": [1_000] * len(closes)},
        index=index,
    )


def test_normalises_yahoo_symbols() -> None:
    assert normalise_us_symbol("BRK.B") == "BRK-B"
    assert normalise_uk_symbol("LON:AZN") == "AZN.L"
    assert normalise_uk_symbol("AZN.L") == "AZN.L"


def test_detects_buy_crossover() -> None:
    result = classify_signal(price_frame([3, 2, 1, 2, 4]), 2, 3)

    assert result is not None
    assert result["signal"] == "BUY"


def test_detects_sell_crossover() -> None:
    result = classify_signal(price_frame([1, 2, 3, 2, 0]), 2, 3)

    assert result is not None
    assert result["signal"] == "SELL"


def test_late_buy_is_marked_wait_for_pullback() -> None:
    closes = [100.0] * 55 + [103, 102, 101, 102, 114]
    result = classify_signal(price_frame(closes), 2, 3, max_rsi=68)

    assert result is not None
    assert result["signal"] == "BUY"
    assert result["entry_status"] == "WAIT_FOR_PULLBACK"
    assert result["rsi_14"] >= 68
    assert result["entry_warning"]


def test_buy_near_resistance_without_persistent_volume_waits_for_confirmation() -> None:
    closes = [10.0] * 70 + [10.3, 10.2, 10.1, 10.2, 10.4]
    result = classify_signal(
        price_frame(closes), 2, 3, max_rsi=100,
        max_short_extension_pct=100, max_five_day_gain_pct=100,
    )

    assert result is not None
    assert result["signal"] == "BUY"
    assert result["near_resistance"] is True
    assert result["persistent_volume_confirmation"] is False
    assert result["entry_status"] == "WAIT_FOR_CONFIRMATION"


def test_returns_none_when_history_is_too_short() -> None:
    assert classify_signal(price_frame([1, 2, 3]), 2, 3) is None


def test_bundled_universe_is_well_formed() -> None:
    path = Path(__file__).parents[1] / "data" / "us_uk_large_mid_mega_cap.csv"
    universe = load_custom_csv(str(path))

    assert len(universe) == 853
    assert universe["ticker"].notna().all()
    assert universe["company"].notna().all()
    assert set(universe["market"]) == {"US", "UK"}
    assert universe.loc[universe["market"] == "UK", "ticker"].str.endswith(".L").all()
    assert not universe.duplicated(["market", "ticker"]).any()


def test_fresh_cache_avoids_market_data_request(tmp_path, monkeypatch) -> None:
    save_cached_ticker(tmp_path, "CACHED", "2y", price_frame([3, 2, 1, 2, 4]))
    universe = pd.DataFrame([
        {"ticker": "CACHED", "company": "Cached Plc", "market": "UK"},
    ])

    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("Fresh cached data should prevent a live request")

    monkeypatch.setattr(screener, "fetch_tickers", unexpected_fetch)
    results, failures = download_and_scan(
        universe, 2, 3, "2y", 25, min_pause=0, max_pause=0,
        cache_dir=tmp_path, cache_max_age_hours=12,
    )

    assert results["ticker"].tolist() == ["CACHED"]
    assert failures.empty
    assert not load_cached_ticker(tmp_path, "CACHED", "2y", 12).empty
    assert load_cached_ticker(tmp_path, "CACHED", "5y", 12).empty


def test_retries_only_failed_symbols(tmp_path, monkeypatch) -> None:
    calls = []
    good = price_frame([3, 2, 1, 2, 4])
    bad = price_frame([1, 2, 3, 2, 0])

    def fake_fetch(tickers, period, threads):
        calls.append((list(tickers), threads))
        if tickers == ["GOOD", "BAD"]:
            return pd.concat({"GOOD": good}, axis=1)
        if tickers == ["BAD"]:
            return bad
        raise AssertionError(f"Unexpected symbols: {tickers}")

    monkeypatch.setattr(screener, "fetch_tickers", fake_fetch)
    universe = pd.DataFrame([
        {"ticker": "GOOD", "company": "Good Inc", "market": "US"},
        {"ticker": "BAD", "company": "Bad Inc", "market": "US"},
    ])
    results, failures = download_and_scan(
        universe, 2, 3, "2y", 25, threads=2, min_pause=0, max_pause=0,
        max_retries=1, retry_base_delay=0, cache_dir=tmp_path,
    )

    assert calls == [(["GOOD", "BAD"], 2), (["BAD"], 2)]
    assert set(results["ticker"]) == {"GOOD", "BAD"}
    assert failures.empty
