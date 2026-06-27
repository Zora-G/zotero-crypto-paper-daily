"""Tests for EprintRetriever."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from omegaconf import open_dict
import feedparser
import pytest

from zotero_arxiv_daily.retriever.eprint_retriever import EprintRetriever


class Entry(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


class MockDateTime:
    def __call__(self, *args, **kwargs):
        return datetime(*args, **kwargs)

    @staticmethod
    def now(tz=None):
        return datetime(2026, 6, 15, 0, 0, tzinfo=timezone.utc)

    @staticmethod
    def strptime(*args, **kwargs):
        return datetime.strptime(*args, **kwargs)


def test_eprint_retriever(config, monkeypatch):
    def _patched_parse(url):
        assert url == "https://eprint.iacr.org/rss/rss.xml"
        return SimpleNamespace(
            bozo=False,
            entries=[
                Entry(
                    title="today match",
                    link="https://eprint.iacr.org/2026/9999",
                    summary="Summary A",
                    comment="Cryptology ePrint report",
                    published_parsed=(2026, 6, 15, 0, 0, 0, 0, 0, 0),
                    tags=[{"term": "Secret-key cryptography"}],
                    authors=[{"name": "Alice"}],
                    links=[{"type": "application/pdf", "href": "https://eprint.iacr.org/2026/9999.pdf"}],
                ),
                Entry(
                    title="today unmatch",
                    link="https://eprint.iacr.org/2026/9998",
                    summary="Summary B",
                    published_parsed=(2026, 6, 15, 0, 0, 1, 0, 0, 0),
                    tags=[SimpleNamespace(term="Quantum cryptography")],
                    authors=[{"name": "Bob"}],
                ),
                Entry(
                    title="category fallback match",
                    link="https://eprint.iacr.org/2026/9996",
                    summary="Summary D",
                    published_parsed=(2026, 6, 15, 0, 0, 2, 0, 0, 0),
                    category="Cryptographic protocols",
                    authors=[{"name": "Dave"}],
                ),
                Entry(
                    title="yesterday old",
                    link="https://eprint.iacr.org/2026/9997",
                    summary="Summary C",
                    published="Sun, 14 Jun 2026 00:00:00 +0000",
                    tags=[{"term": "Secret-key cryptography"}],
                    authors=[{"name": "Carol"}],
                ),
            ],
        )

    monkeypatch.setattr(feedparser, "parse", _patched_parse)
    monkeypatch.setattr(
        "zotero_arxiv_daily.retriever.eprint_retriever.datetime",
        MockDateTime(),
    )

    with open_dict(config.source):
        config.source.eprint = {"category": ["Secret-key cryptography", "Cryptographic protocols"]}

    retriever = EprintRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == 2
    assert {p.title for p in papers} == {"today match", "category fallback match"}
    assert papers[0].source == "eprint"
    assert any(p.url == "https://eprint.iacr.org/2026/9999" for p in papers)
    assert any(p.pdf_url == "https://eprint.iacr.org/2026/9999.pdf" for p in papers)
    first_paper = next(p for p in papers if p.title == "today match")
    assert first_paper.source_note == "Cryptology ePrint report"


def test_eprint_days_back(config, monkeypatch):
    def _patched_parse(url):
        assert url == "https://eprint.iacr.org/rss/rss.xml"
        Entry = SimpleNamespace
        return SimpleNamespace(
            bozo=False,
            entries=[
                Entry(
                    title="today match",
                    link="https://eprint.iacr.org/2026/9999",
                    summary="Summary A",
                    published_parsed=(2026, 6, 15, 0, 0, 0, 0, 0, 0),
                    tags=[{"term": "Secret-key cryptography"}],
                    authors=[{"name": "Alice"}],
                    links=[{"type": "application/pdf", "href": "https://eprint.iacr.org/2026/9999.pdf"}],
                ),
                Entry(
                    title="yesterday match",
                    link="https://eprint.iacr.org/2026/9998",
                    summary="Summary B",
                    published_parsed=(2026, 6, 14, 12, 0, 0, 0, 0, 0),
                    tags=[{"term": "Secret-key cryptography"}],
                    authors=[{"name": "Bob"}],
                ),
            ],
        )

    monkeypatch.setattr(feedparser, "parse", _patched_parse)
    monkeypatch.setattr(
        "zotero_arxiv_daily.retriever.eprint_retriever.datetime",
        MockDateTime(),
    )

    with open_dict(config.source):
        config.source.eprint = {"category": ["Secret-key cryptography"], "days_back": 2}

    retriever = EprintRetriever(config)
    papers = retriever.retrieve_papers()

    assert {p.title for p in papers} == {"today match", "yesterday match"}


def test_eprint_requires_category(config):
    with open_dict(config.source):
        config.source.eprint = {"category": None}
    with pytest.raises(ValueError, match="category must be specified"):
        EprintRetriever(config)
