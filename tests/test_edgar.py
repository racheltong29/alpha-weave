from unittest.mock import MagicMock, patch

import pytest

from alphaweave.data import edgar


@pytest.fixture(autouse=True)
def _clear_cik_cache():
    edgar._ticker_to_cik_map.cache_clear()
    yield
    edgar._ticker_to_cik_map.cache_clear()


def _mock_response(json_data=None, text_data=None):
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.text = text_data
    resp.raise_for_status.return_value = None
    return resp


def test_cik_for_ticker_looks_up_and_zero_pads():
    tickers_json = {"0": {"ticker": "AAPL", "cik_str": 320193}}
    with patch.object(
        edgar.requests, "get", return_value=_mock_response(json_data=tickers_json)
    ):
        assert edgar._cik_for_ticker("aapl") == "0000320193"


def test_cik_for_unknown_ticker_raises():
    with patch.object(
        edgar.requests, "get", return_value=_mock_response(json_data={})
    ):
        with pytest.raises(ValueError):
            edgar._cik_for_ticker("NOPE")


def test_list_8k_filings_filters_to_item_2_02_in_range():
    submissions = {
        "filings": {
            "recent": {
                "form": ["8-K", "8-K", "10-Q"],
                "filingDate": ["2023-02-02", "2023-05-04", "2023-03-01"],
                "items": ["2.02,9.01", "5.02", ""],
                "accessionNumber": ["0000320193-23-000001", "0000320193-23-000002", "0000320193-23-000003"],
            }
        }
    }
    with patch.object(
        edgar.requests, "get", return_value=_mock_response(json_data=submissions)
    ):
        filings = edgar._list_8k_filings("0000320193", "2023-01-01", "2023-06-01")

    assert len(filings) == 1
    assert filings[0]["filing_date"] == "2023-02-02"


def test_fetch_earnings_exhibit_text_finds_ex99_document():
    index_json = {
        "directory": {
            "item": [
                {"name": "aapl-20230101.htm"},
                {"name": "a8-kex991q1.htm"},
            ]
        }
    }
    html = "<html><body><p>Apple reports record revenue</p></body></html>"

    def fake_get(url, headers=None, timeout=None):
        if url.endswith("index.json"):
            return _mock_response(json_data=index_json)
        return _mock_response(text_data=html)

    with patch.object(edgar.requests, "get", side_effect=fake_get):
        text = edgar._fetch_earnings_exhibit_text("0000320193", "000032019323000001")

    assert "record revenue" in text


def test_fetch_earnings_exhibit_text_returns_none_without_exhibit():
    index_json = {"directory": {"item": [{"name": "aapl-20230101.htm"}]}}
    with patch.object(
        edgar.requests, "get", return_value=_mock_response(json_data=index_json)
    ):
        assert edgar._fetch_earnings_exhibit_text("0000320193", "acc") is None


def test_fetch_earnings_release_text_returns_indexed_frame():
    tickers_json = {"0": {"ticker": "AAPL", "cik_str": 320193}}
    submissions = {
        "filings": {
            "recent": {
                "form": ["8-K"],
                "filingDate": ["2023-02-02"],
                "items": ["2.02"],
                "accessionNumber": ["0000320193-23-000001"],
            }
        }
    }
    index_json = {"directory": {"item": [{"name": "a8-kex991.htm"}]}}
    html = "<html><body>Earnings text</body></html>"

    def fake_get(url, headers=None, timeout=None):
        if "company_tickers.json" in url:
            return _mock_response(json_data=tickers_json)
        if "data.sec.gov/submissions" in url:
            return _mock_response(json_data=submissions)
        if url.endswith("index.json"):
            return _mock_response(json_data=index_json)
        return _mock_response(text_data=html)

    with patch.object(edgar.requests, "get", side_effect=fake_get):
        frame = edgar.fetch_earnings_release_text(
            ["AAPL"], "2023-01-01", "2023-06-01", request_delay_seconds=0
        )

    assert list(frame.index)[0] == (pytest_timestamp("2023-02-02"), "AAPL")
    assert "Earnings text" in frame.iloc[0]["text"]


def pytest_timestamp(date_str):
    import pandas as pd

    return pd.Timestamp(date_str)
