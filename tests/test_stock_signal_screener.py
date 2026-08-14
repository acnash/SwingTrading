from pathlib import Path

import pandas as pd

from stock_signal_screener import (
    classify_signal,
    load_custom_csv,
    normalise_uk_symbol,
    normalise_us_symbol,
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
