#!/usr/bin/env python3
"""Regenerate the bundled US and UK large/mid-cap custom universe."""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stock_signal_screener import normalise_uk_symbol, normalise_us_symbol, read_html_tables


SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FTSE100_URL = "https://en.wikipedia.org/wiki/FTSE_100_Index"
FTSE250_URL = "https://en.wikipedia.org/wiki/FTSE_250_Index"
OUTPUT = PROJECT_ROOT / "data" / "us_uk_large_mid_mega_cap.csv"


def find_company_ticker_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
    for table in tables:
        columns = {str(column).lower(): column for column in table.columns}
        if "company" in columns and "ticker" in columns:
            return table[[columns["ticker"], columns["company"]]].rename(
                columns={columns["ticker"]: "ticker", columns["company"]: "company"}
            )
    raise RuntimeError("Could not locate a Company/Ticker constituent table")


def build_universe() -> pd.DataFrame:
    sp500_table = read_html_tables(SP500_URL)[0][["Symbol", "Security"]].copy()
    sp500_table.columns = ["ticker", "company"]
    sp500_table["ticker"] = sp500_table["ticker"].map(normalise_us_symbol)
    sp500_table["market"] = "US"

    ftse100 = find_company_ticker_table(read_html_tables(FTSE100_URL))
    ftse250 = find_company_ticker_table(read_html_tables(FTSE250_URL))
    london = pd.concat([ftse100, ftse250], ignore_index=True).dropna(subset=["ticker", "company"])
    london["ticker"] = london["ticker"].astype(str).str.replace(".", "-", regex=False)
    london["ticker"] = london["ticker"].map(normalise_uk_symbol)
    london["market"] = "UK"

    universe = pd.concat([sp500_table, london], ignore_index=True)
    universe = universe.drop_duplicates(subset=["market", "ticker"])
    return universe[["ticker", "company", "market"]].sort_values(
        ["market", "company", "ticker"], ignore_index=True
    )


def main() -> None:
    universe = build_universe()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(universe)} rows to {OUTPUT}")
    print(universe.groupby("market").size().to_string())


if __name__ == "__main__":
    main()
