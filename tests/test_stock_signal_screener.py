from pathlib import Path

import pandas as pd
import pytest
import stock_signal_screener as screener

from stock_signal_screener import (
    add_vfi_columns,
    build_universe,
    classify_signal,
    download_and_scan,
    get_nasdaq_mid_large_mega,
    is_nasdaq_primary_equity,
    load_custom_csv,
    load_cached_ticker,
    normalise_uk_symbol,
    normalise_us_symbol,
    parse_market_cap,
    save_cached_ticker,
    vfi_confirmation_score,
    write_three_ai_prompts,
)


def price_frame(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {"Close": closes, "Volume": [1_000] * len(closes)},
        index=index,
    )


def ohlcv_frame(closes: list[float], volumes: list[float]) -> pd.DataFrame:
    frame = price_frame(closes)
    frame["High"] = frame["Close"] * 1.01
    frame["Low"] = frame["Close"] * 0.99
    frame["Volume"] = volumes
    return frame


def test_vfi_buy_index_detects_accumulation() -> None:
    closes = [100 + index * 0.25 for index in range(220)]
    volumes = [1_000 + index * 5 for index in range(220)]
    with_vfi = add_vfi_columns(ohlcv_frame(closes, volumes))
    assert with_vfi["VFI"].notna().sum() > 20

    vfi = [-8.0] * 18 + [-6.0, -4.0, -2.0, 0.5, 2.0, 4.0, 6.0]
    signal = [-5.0] * 21 + [1.0, 1.5, 2.0, 3.0]
    score, classification = vfi_confirmation_score(pd.DataFrame({
        "VFI": vfi,
        "VFI_signal": signal,
    }))

    assert score is not None
    assert score >= 65
    assert classification in {"SUPPORTS_BUY", "STRONG_ACCUMULATION"}


def test_normalises_yahoo_symbols() -> None:
    assert normalise_us_symbol("BRK.B") == "BRK-B"
    assert normalise_uk_symbol("LON:AZN") == "AZN.L"
    assert normalise_uk_symbol("AZN.L") == "AZN.L"


def test_parses_nasdaq_market_caps() -> None:
    assert parse_market_cap("2,500,000,000.00") == 2_500_000_000
    assert parse_market_cap("$10,000") == 10_000
    assert pd.isna(parse_market_cap("N/A"))


@pytest.mark.parametrize(
    ("name", "industry", "expected"),
    [
        ("Example Inc. Common Stock", "Software", True),
        ("Example plc American Depositary Shares", "Software", True),
        ("Pipeline L.P. Common Units representing Limited Partner Interests", "Oil & Gas", True),
        ("Example Inc. Series A Preferred Stock", "Finance", False),
        ("Example Inc. Warrants", "Software", False),
        ("Example Inc. Senior Notes due 2030", "Finance", False),
        ("Example Inc. 8.00% Notes due 2028", "Finance", False),
        ("Acquisition Corp. Class A Common Stock", "Blank Checks", False),
    ],
)
def test_identifies_nasdaq_primary_equities(
    name: str, industry: str, expected: bool,
) -> None:
    assert is_nasdaq_primary_equity(name, industry) is expected


def test_get_nasdaq_mid_large_mega_filters_rows(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "data": {
                    "rows": [
                        {
                            "symbol": "MID",
                            "name": "Mid Company Common Stock",
                            "marketCap": "2,000,000,000",
                            "industry": "Software",
                        },
                        {
                            "symbol": "SMALL",
                            "name": "Small Company Common Stock",
                            "marketCap": "1,999,999,999",
                            "industry": "Software",
                        },
                        {
                            "symbol": "MIDP",
                            "name": "Mid Company Preferred Stock",
                            "marketCap": "4,000,000,000",
                            "industry": "Software",
                        },
                    ]
                }
            }

    monkeypatch.setattr(screener.requests, "get", lambda *args, **kwargs: FakeResponse())
    universe = get_nasdaq_mid_large_mega()

    assert universe.to_dict("records") == [
        {"ticker": "MID", "company": "Mid Company Common Stock", "market": "US"}
    ]


def test_build_universe_includes_nasdaq_and_deduplicates(monkeypatch) -> None:
    monkeypatch.setattr(
        screener,
        "get_sp500",
        lambda: pd.DataFrame([
            {"ticker": "SHARED", "company": "Shared Inc.", "market": "US"},
        ]),
    )
    monkeypatch.setattr(
        screener,
        "get_ftse100",
        lambda: pd.DataFrame([
            {"ticker": "UKCO.L", "company": "UK Co", "market": "UK"},
        ]),
    )
    monkeypatch.setattr(
        screener,
        "get_nasdaq_mid_large_mega",
        lambda: pd.DataFrame([
            {"ticker": "SHARED", "company": "Shared Inc. Common Stock", "market": "US"},
            {"ticker": "NASDAQ", "company": "Nasdaq Mid Cap", "market": "US"},
        ]),
    )

    universe = build_universe(True, True, None, include_nasdaq=True)

    assert set(universe["ticker"]) == {"NASDAQ", "SHARED", "UKCO.L"}
    assert not universe.duplicated("ticker").any()


def test_detects_buy_crossover() -> None:
    result = classify_signal(price_frame([3, 2, 1, 2, 4]), 2, 3)

    assert result is not None
    assert result["signal"] == "BUY"


def test_detects_sell_crossover() -> None:
    result = classify_signal(price_frame([1, 2, 3, 2, 0]), 2, 3)

    assert result is not None
    assert result["signal"] == "SELL"


def test_weak_vfi_moves_buy_to_volume_confirmation(monkeypatch) -> None:
    monkeypatch.setattr(
        screener, "vfi_confirmation_score",
        lambda work: (35, "WEAK_VOLUME_SUPPORT"),
    )
    result = classify_signal(
        price_frame([1.0] * 200 + [3, 2, 1, 2, 4]), 2, 3,
        max_rsi=1000, max_short_extension_pct=1000,
        max_five_day_gain_pct=1000, resistance_proximity_pct=0,
    )

    assert result is not None
    assert result["signal"] == "BUY"
    assert result["fifty_two_week_high"] is None
    assert result["fifty_two_week_low"] is None
    assert result["at_52_week_peak"] is False
    assert result["vfi_buy_index"] == 35
    assert result["entry_status"] == "WAIT_FOR_VOLUME_CONFIRMATION"


def test_buy_below_200_day_sma_waits_for_reclaim() -> None:
    closes = [100.0] * 252 + [3, 2, 1, 2, 4]
    result = classify_signal(
        price_frame(closes), 2, 3, max_rsi=1000,
        max_short_extension_pct=1000, max_five_day_gain_pct=1000,
        resistance_proximity_pct=0, min_vfi_buy_score=0,
    )

    assert result is not None
    assert result["signal"] == "BUY"
    assert result["above_200_sma"] is False
    assert result["entry_status"] == "WAIT_FOR_200_SMA_RECLAIM"
    assert "below 200-day SMA" in result["entry_warning"]


def test_buy_without_200_day_sma_waits_for_confirmation() -> None:
    result = classify_signal(
        price_frame([3, 2, 1, 2, 4]), 2, 3, max_rsi=1000,
        max_short_extension_pct=1000, max_five_day_gain_pct=1000,
        resistance_proximity_pct=0, min_vfi_buy_score=0,
    )

    assert result is not None
    assert result["signal"] == "BUY"
    assert result["above_200_sma"] is None
    assert result["entry_status"] == "WAIT_FOR_200_SMA_RECLAIM"
    assert "200-day SMA unavailable" in result["entry_warning"]


def test_late_buy_is_marked_wait_for_pullback() -> None:
    closes = [100.0] * 55 + [103, 102, 101, 102, 114]
    result = classify_signal(price_frame(closes), 2, 3, max_rsi=68)

    assert result is not None
    assert result["signal"] == "BUY"
    assert result["entry_status"] == "WAIT_FOR_PULLBACK"
    assert result["rsi_14"] >= 68
    assert result["entry_warning"]


def test_buy_within_one_percent_of_52_week_high_waits_for_pullback() -> None:
    closes = [1.0] * 252 + [3, 2, 1, 2, 4]
    frame = ohlcv_frame(closes, [1_000] * len(closes))
    result = classify_signal(
        frame, 2, 3, max_rsi=1000,
        max_short_extension_pct=1000, max_five_day_gain_pct=1000,
        resistance_proximity_pct=0, min_vfi_buy_score=0,
    )

    assert result is not None
    assert result["signal"] == "BUY"
    assert result["fifty_two_week_high"] == pytest.approx(4.04)
    assert result["fifty_two_week_low"] == pytest.approx(0.99)
    assert result["distance_from_52_week_high_pct"] == pytest.approx(0.990099)
    assert result["at_52_week_peak"] is True
    assert result["entry_status"] == "WAIT_FOR_PULLBACK"
    assert "52-week high" in result["entry_warning"]


def test_buy_above_200_day_and_more_than_one_percent_below_high_is_actionable() -> None:
    closes = [95.0] * 10 + [101.0] + [95.0] * 241 + [98, 97, 96, 97, 99]
    frame = ohlcv_frame(closes, [1_000] * len(closes))
    result = classify_signal(
        frame, 2, 3, max_rsi=1000,
        max_short_extension_pct=1000, max_five_day_gain_pct=1000,
        resistance_proximity_pct=0, min_vfi_buy_score=0,
    )

    assert result is not None
    assert result["signal"] == "BUY"
    assert result["distance_from_52_week_high_pct"] > 1.0
    assert result["at_52_week_peak"] is False
    assert result["above_200_sma"] is True
    assert result["entry_status"] == "ACTIONABLE_BUY"


def test_buy_near_resistance_without_persistent_volume_waits_for_confirmation() -> None:
    closes = [10.0] * 200 + [10.3, 10.2, 10.1, 10.2, 10.4]
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

    assert len(universe) >= 1_000
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


def test_writes_three_independent_review_prompts(tmp_path) -> None:
    buys = pd.DataFrame([{
        "ticker": "AAA.L", "company": "AAA Plc", "market": "UK",
        "date": "2026-08-14", "close": 100.0, "short_sma": 99.0,
        "medium_sma": 98.0, "cross_gap_pct": 1.0, "sma_200": 95.0,
        "above_200_sma": True, "volume_ratio": 1.3, "rsi_14": 55.0,
        "short_sma_extension_pct": 1.0, "five_day_gain_pct": 2.0,
        "prior_resistance": 105.0, "resistance_distance_pct": 5.0,
        "fifty_two_week_high": 110.0, "fifty_two_week_low": 70.0,
        "distance_from_52_week_high_pct": 9.1,
        "position_in_52_week_range_pct": 75.0, "at_52_week_peak": False,
        "persistent_volume_confirmation": True, "entry_status": "ACTIONABLE_BUY",
        "entry_warning": "",
    }])

    write_three_ai_prompts(buys, tmp_path)

    prompts = sorted(tmp_path.glob("buy_signal_review_prompt_*.txt"))
    assert len(prompts) == 3
    for number, prompt in enumerate(prompts, start=1):
        text = prompt.read_text(encoding="utf-8")
        assert f"INDEPENDENT REVIEW {number} OF 3" in text
        assert "Do not inspect, copy, reconcile or average" in text
