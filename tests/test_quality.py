"""Tests for OpenAlex-based quality enrichment."""

from omegaconf import open_dict

from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily.quality import OpenAlexQualityEnricher


def test_openalex_quality_enricher_populates_metadata(config, monkeypatch):
    with open_dict(config.executor):
        config.executor.quality_boost = True
        config.executor.quality_search_results = 5
        config.executor.quality_max_authors = 2
        config.executor.quality_min_title_similarity = 0.6

    enricher = OpenAlexQualityEnricher(config)

    work_payload = {
        "results": [
            {
                "title": "Unrelated title",
                "cited_by_count": 999,
            },
            {
                "title": "Sample Paper Title",
                "cited_by_count": 42,
                "primary_location": {
                    "source": {
                        "display_name": "CRYPTO",
                        "type": "conference",
                    }
                },
                "authorships": [
                    {"author": {"id": "https://openalex.org/A1"}},
                    {"author": {"id": "https://openalex.org/A2"}},
                ],
            },
        ]
    }
    def _fake_fetch_json(url):
        if "works?" in url:
            return work_payload
        if "/authors/A1" in url:
            return {"summary_stats": {"h_index": 10}}
        if "/authors/A2" in url:
            return {"summary_stats": {"h_index": 6}}
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(enricher, "_fetch_json", _fake_fetch_json)

    paper = make_sample_paper()
    enricher.enrich_paper(paper)

    assert paper.citation_count == 42
    assert paper.venue_name == "CRYPTO"
    assert paper.venue_type == "conference"
    assert paper.author_h_index == 8.0
