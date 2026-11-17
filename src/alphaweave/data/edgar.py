"""Point-in-time text ingestion from SEC EDGAR: 8-K earnings-release
exhibits (item 2.02, "Results of Operations and Financial Condition").

This is the free, no-API-key alternative to a paid transcript provider. A
filing's text is only known as of its `filingDate`, so indexing by that date
keeps this ingestion point-in-time correct the same way `ingest.py`'s price
data is.
"""

import time
from functools import lru_cache

import pandas as pd
import requests
from bs4 import BeautifulSoup

_SEC_CONTACT = "AlphaWeave research rachelto@andrew.cmu.edu"
_HEADERS = {"User-Agent": _SEC_CONTACT}


@lru_cache(maxsize=1)
def _ticker_to_cik_map() -> dict[str, str]:
    """SEC's ticker->CIK lookup table, zero-padded to 10 digits."""
    resp = requests.get(
        "https://www.sec.gov/files/company_tickers.json", headers=_HEADERS, timeout=10
    )
    resp.raise_for_status()
    return {
        row["ticker"].upper(): str(row["cik_str"]).zfill(10)
        for row in resp.json().values()
    }


def _cik_for_ticker(ticker: str) -> str:
    cik = _ticker_to_cik_map().get(ticker.upper())
    if cik is None:
        raise ValueError(f"No SEC CIK found for ticker '{ticker}'")
    return cik


def _list_8k_filings(cik: str, start: str, end: str) -> list[dict]:
    resp = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=_HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    recent = resp.json()["filings"]["recent"]
    filings = []
    for i, form in enumerate(recent["form"]):
        filing_date = recent["filingDate"][i]
        if form != "8-K" or not (start <= filing_date <= end):
            continue
        items = recent.get("items", [""] * len(recent["form"]))[i]
        if "2.02" not in items.split(","):
            continue
        filings.append(
            {
                "filing_date": filing_date,
                "accession_number": recent["accessionNumber"][i].replace("-", ""),
            }
        )
    return filings


def _fetch_earnings_exhibit_text(cik: str, accession_number: str) -> str | None:
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{accession_number}/index.json"
    )
    resp = requests.get(index_url, headers=_HEADERS, timeout=10)
    resp.raise_for_status()
    documents = resp.json()["directory"]["item"]
    exhibit = next(
        (
            doc["name"]
            for doc in documents
            if "ex99" in doc["name"].lower() or "ex-99" in doc["name"].lower()
        ),
        None,
    )
    if exhibit is None:
        return None
    doc_url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
        f"{accession_number}/{exhibit}"
    )
    doc_resp = requests.get(doc_url, headers=_HEADERS, timeout=10)
    doc_resp.raise_for_status()
    soup = BeautifulSoup(doc_resp.text, "html.parser")
    return soup.get_text(" ", strip=True)


def fetch_earnings_release_text(
    tickers: list[str], start: str, end: str, request_delay_seconds: float = 0.15
) -> pd.DataFrame:
    """Fetch 8-K item-2.02 earnings-release text for `tickers` between
    `start` and `end` (YYYY-MM-DD).

    Returns a long frame indexed by (date, ticker) with a 'text' column,
    matching what `factors.signals.compute_signal` expects. `request_delay_seconds`
    throttles requests to stay well under SEC EDGAR's fair-use rate limit.
    """
    rows = []
    for ticker in tickers:
        cik = _cik_for_ticker(ticker)
        for filing in _list_8k_filings(cik, start, end):
            time.sleep(request_delay_seconds)
            text = _fetch_earnings_exhibit_text(cik, filing["accession_number"])
            if text:
                rows.append(
                    {
                        "date": pd.Timestamp(filing["filing_date"]),
                        "ticker": ticker,
                        "text": text,
                    }
                )
            time.sleep(request_delay_seconds)
    frame = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    return frame
