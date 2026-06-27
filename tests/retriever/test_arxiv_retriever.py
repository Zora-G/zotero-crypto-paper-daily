"""Tests for ArxivRetriever."""

import time
from datetime import date
from types import SimpleNamespace

import feedparser
from omegaconf import open_dict

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_arxiv_retriever(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    with open_dict(config.source.arxiv):
        config.source.arxiv.days_back = 1
    monkeypatch.setattr(arxiv_retriever, "_utc_today", lambda: date(2025, 8, 20))

    # The RSS fixture gives us paper IDs.  After feedparser, the code calls
    # arxiv.Client().results(search) which makes real HTTP requests.  We mock
    # the arxiv Client so the test stays offline.
    new_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]
    paper_ids = [e.id.removeprefix("oai:arXiv.org:") for e in new_entries]

    # Build fake ArxivResult-like objects matching each RSS entry
    fake_results = []
    for idx, entry in enumerate(new_entries):
        pid = entry.id.removeprefix("oai:arXiv.org:")
        comment = "To appear at Crypto 2027" if idx == 0 else None
        fake_results.append(SimpleNamespace(
            title=entry.title,
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url=f"https://arxiv.org/pdf/{pid}",
            entry_id=f"https://arxiv.org/abs/{pid}",
            source_url=lambda pid=pid: f"https://arxiv.org/e-print/{pid}",
            comment=comment,
        ))

    class FakeClient:
        def __init__(self, **kw):
            pass
        def results(self, search):
            return iter(fake_results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    # Skip file downloads in convert_to_paper
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert set(p.title for p in papers) == set(e.title for e in new_entries)
    assert papers[0].source_note == "To appear at Crypto 2027"


def test_arxiv_retriever_falls_back_to_web_new_listing_when_rss_empty(config, monkeypatch):
    target_date = date(2026, 6, 15)
    with open_dict(config.source.arxiv):
        config.source.arxiv.days_back = 1
    today_primary = SimpleNamespace(
        title="Today primary",
        entry_id="https://arxiv.org/abs/2606.00001v1",
    )
    class FakeClient:
        def __init__(self, **kw):
            pass

        def results(self, search):
            return iter([today_primary])

    html = """
    <h3>Showing new listings for Monday, 15 June 2026</h3>
    <h3>New submissions (showing 1 of 1 entries)</h3>
    <a href ="/abs/2606.00001" title="Abstract" id="2606.00001">arXiv:2606.00001</a>
    <h3>Cross submissions (showing 1 of 1 entries)</h3>
    <a href ="/abs/2606.00002" title="Abstract" id="2606.00002">arXiv:2606.00002</a>
    <h3>Replacement submissions (showing 1 of 1 entries)</h3>
    <a href ="/abs/2606.00003" title="Abstract" id="2606.00003">arXiv:2606.00003</a>
    """

    def fake_get(url, **kwargs):
        return SimpleNamespace(text=html, raise_for_status=lambda: None)

    empty_feed = SimpleNamespace(feed=SimpleNamespace(title="cs.AI updates"), entries=[])
    monkeypatch.setattr(feedparser, "parse", lambda url: empty_feed)
    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)
    monkeypatch.setattr(arxiv_retriever.requests, "get", fake_get)
    monkeypatch.setattr(arxiv_retriever, "_utc_today", lambda: target_date)

    retriever = ArxivRetriever(config)
    raw_papers = retriever._retrieve_raw_papers()

    assert raw_papers == [today_primary]


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=1, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=1, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]
