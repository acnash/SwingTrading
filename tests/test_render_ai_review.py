import json

import pytest

from render_ai_review import aggregate_reviews


def write_review(path, scores):
    path.write_text(json.dumps([
        {
            "ticker": ticker,
            "company": f"{ticker} Company",
            "market": "UK",
            "signal_date": "2026-08-14",
            "evidence_support_pct": score,
        }
        for ticker, score in scores.items()
    ]), encoding="utf-8")


def test_aggregates_exactly_three_reviews(tmp_path):
    paths = [tmp_path / f"review_{number}.json" for number in range(1, 4)]
    write_review(paths[0], {"AAA.L": 80, "BBB.L": 40})
    write_review(paths[1], {"AAA.L": 70, "BBB.L": 50})
    write_review(paths[2], {"AAA.L": 75, "BBB.L": 45})

    result = aggregate_reviews(paths)

    assert result["Ticker"].tolist() == ["AAA.L", "BBB.L"]
    assert result["Mean %"].tolist() == [75.0, 45.0]
    assert result["Classification"].tolist() == [
        "SUPPORTS BUY SIGNAL", "WEAKENS BUY SIGNAL"
    ]
    assert result["Research action"].tolist() == [
        "BUY CANDIDATE", "REJECT BUY / REVIEW EXIT"
    ]


def test_rejects_wrong_review_count(tmp_path):
    paths = [tmp_path / "one.json", tmp_path / "two.json"]
    for path in paths:
        write_review(path, {"AAA.L": 60})

    with pytest.raises(ValueError, match="Exactly 3"):
        aggregate_reviews(paths)


def test_rejects_mismatched_tickers(tmp_path):
    paths = [tmp_path / f"review_{number}.json" for number in range(1, 4)]
    write_review(paths[0], {"AAA.L": 60})
    write_review(paths[1], {"AAA.L": 60})
    write_review(paths[2], {"BBB.L": 60})

    with pytest.raises(ValueError, match="ticker set differs"):
        aggregate_reviews(paths)
