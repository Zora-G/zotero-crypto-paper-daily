import json
import re
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from loguru import logger

from .protocol import Paper

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _normalize_title(title: str) -> str:
    return " ".join(_NORMALIZE_RE.sub(" ", title.lower()).split())


def _title_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, _normalize_title(left), _normalize_title(right)).ratio()


class OpenAlexQualityEnricher:
    """Best-effort OpenAlex metadata fetcher for quality-related signals."""

    def __init__(self, config):
        executor_cfg = getattr(config, "executor", None)
        email_cfg = getattr(config, "email", None)
        self.enabled = bool(getattr(executor_cfg, "quality_boost", False))
        self.search_results = int(getattr(executor_cfg, "quality_search_results", 5))
        self.max_authors = int(getattr(executor_cfg, "quality_max_authors", 1))
        self.timeout = float(getattr(executor_cfg, "quality_timeout", 10))
        self.min_title_similarity = float(getattr(executor_cfg, "quality_min_title_similarity", 0.6))
        self.mailto = getattr(email_cfg, "sender", None) or getattr(email_cfg, "receiver", None)
        self.user_agent = "zotero-arxiv-daily/quality-enricher"
        self._work_cache: dict[str, dict[str, Any] | None] = {}
        self._author_cache: dict[str, float | None] = {}

    def enrich_papers(self, papers: list[Paper]) -> list[Paper]:
        if not self.enabled:
            return papers

        for paper in papers:
            self.enrich_paper(paper)
        return papers

    def enrich_paper(self, paper: Paper) -> Paper:
        if not self.enabled or not paper.title:
            return paper

        work = self._match_work(paper.title)
        if not work:
            return paper

        paper.citation_count = self._safe_int(work.get("cited_by_count"))

        primary_location = work.get("primary_location") or {}
        source = primary_location.get("source") or {}
        paper.venue_name = source.get("display_name")
        paper.venue_type = source.get("type")

        h_indexes = []
        for authorship in (work.get("authorships") or [])[: self.max_authors]:
            author = authorship.get("author") or {}
            author_id = author.get("id")
            if not author_id:
                continue
            h_index = self._fetch_author_h_index(str(author_id))
            if h_index is not None:
                h_indexes.append(h_index)

        if h_indexes:
            paper.author_h_index = sum(h_indexes) / len(h_indexes)

        return paper

    def _fetch_json(self, url: str) -> dict[str, Any]:
        request = Request(url, headers={"User-Agent": self.user_agent, "Accept": "application/json"})
        with urlopen(request, timeout=self.timeout) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload)

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        try:
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            return float(value)
        except Exception:
            return None

    def _match_work(self, title: str) -> dict[str, Any] | None:
        normalized_title = _normalize_title(title)
        if normalized_title in self._work_cache:
            return self._work_cache[normalized_title]

        params = {
            "search": title,
            "per-page": self.search_results,
        }
        if self.mailto:
            params["mailto"] = self.mailto
        url = f"https://api.openalex.org/works?{urlencode(params)}"

        try:
            payload = self._fetch_json(url)
        except Exception as exc:
            logger.debug(f"OpenAlex work lookup failed for {title!r}: {exc}")
            self._work_cache[normalized_title] = None
            return None

        best_item = None
        best_score = 0.0
        for item in payload.get("results", []):
            candidate_title = item.get("title") or item.get("display_name") or ""
            if not candidate_title:
                continue
            score = _title_similarity(title, candidate_title)
            if score > best_score:
                best_score = score
                best_item = item

        if best_score < self.min_title_similarity:
            logger.debug(f"OpenAlex title match too weak for {title!r}: {best_score:.3f}")
            best_item = None

        self._work_cache[normalized_title] = best_item
        return best_item

    def _fetch_author_h_index(self, author_id: str) -> float | None:
        author_key = author_id.rsplit("/", 1)[-1]
        if author_key in self._author_cache:
            return self._author_cache[author_key]

        params = {}
        if self.mailto:
            params["mailto"] = self.mailto
        url = f"https://api.openalex.org/authors/{author_key}"
        if params:
            url = f"{url}?{urlencode(params)}"

        try:
            payload = self._fetch_json(url)
        except Exception as exc:
            logger.debug(f"OpenAlex author lookup failed for {author_id!r}: {exc}")
            self._author_cache[author_key] = None
            return None

        summary_stats = payload.get("summary_stats") or {}
        h_index = self._safe_float(summary_stats.get("h_index"))
        self._author_cache[author_key] = h_index
        return h_index
