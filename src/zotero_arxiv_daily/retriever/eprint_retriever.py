import re
from datetime import datetime, timedelta, timezone
from typing import Any
from html import unescape
from collections import Counter

import feedparser
from loguru import logger

from .base import BaseRetriever, register_retriever
from ..protocol import Paper


def _entry_get(entry: Any, key: str, default=None):
    try:
        return entry.get(key, default)
    except AttributeError:
        return getattr(entry, key, default)


@register_retriever("eprint")
class EprintRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.retriever_config.category is None:
            raise ValueError(f"category must be specified for {self.name}")

    @staticmethod
    def _extract_time(entry: Any) -> datetime | None:
        parsed_time = None
        if parsed := getattr(entry, "published_parsed", None):
            try:
                parsed_time = datetime(*parsed[:6], tzinfo=timezone.utc)
            except Exception:
                parsed_time = None
        if parsed_time is None:
            parsed_time = getattr(entry, "updated_parsed", None)
            if parsed_time is not None:
                try:
                    parsed_time = datetime(*parsed_time[:6], tzinfo=timezone.utc)
                except Exception:
                    parsed_time = None

        raw_time = getattr(entry, "published", None) or getattr(entry, "updated", None)
        if parsed_time is None and raw_time is not None:
            for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
                try:
                    parsed_time = datetime.strptime(raw_time, fmt)
                    if parsed_time.tzinfo is None:
                        parsed_time = parsed_time.replace(tzinfo=timezone.utc)
                    return parsed_time
                except Exception:
                    continue
        return parsed_time

    @staticmethod
    def _extract_categories(entry: Any) -> set[str]:
        entry_categories = set()
        for tag in getattr(entry, "tags", []):
            if isinstance(tag, dict):
                term = tag.get("term")
            else:
                term = getattr(tag, "term", None)
            if term:
                entry_categories.add(str(term).strip().lower())

        category_value = getattr(entry, "category", None)
        if category_value is None:
            category_value = getattr(entry, "dc_subject", None)
        if category_value is not None:
            entry_categories.update(
                str(v).strip().lower() for v in ([category_value] if not isinstance(category_value, list) else category_value)
            )
        return entry_categories

    @staticmethod
    def _category_hit(entry_categories: set[str], configured: set[str]) -> bool:
        if not configured:
            return True
        if not entry_categories:
            return False
        for cfg in configured:
            for entry_category in entry_categories:
                if cfg in entry_category or entry_category in cfg or entry_category == cfg:
                    return True
        return False

    def _retrieve_raw_papers(self) -> list[Any]:
        response = feedparser.parse("https://eprint.iacr.org/rss/rss.xml")
        logger.info(f"Loaded ePrint RSS with {len(response.entries)} entries.")
        if len(response.entries) == 0:
            if response.bozo:
                logger.warning(f"Failed to parse ePrint RSS: {response.bozo_exception}")
                return []
            logger.warning("No papers found in ePrint RSS.")
            return []

        category_set = {str(c).strip().lower() for c in self.retriever_config.category}
        days_back = int(getattr(self.retriever_config, "days_back", 1))
        if days_back < 1:
            days_back = 1

        today = datetime.now(timezone.utc).date()
        target_dates = {today}
        for i in range(1, days_back):
            target_dates.add(today - timedelta(days=i))
        logger.info(f"Target ePrint dates: {sorted(target_dates)}")

        date_hits: Counter[str] = Counter()
        papers = []
        matched_today_count = 0
        category_matched_count = 0
        matched_entries: list[tuple[datetime, Any]] = []
        for entry in response.entries:
            entry_time = self._extract_time(entry)
            if entry_time is None:
                logger.warning(f"Failed to parse ePrint time from {_entry_get(entry, 'link')}")
                continue
            date_key = entry_time.strftime("%Y-%m-%d")
            date_hits[date_key] += 1
            if entry_time.date() not in target_dates:
                continue

            matched_today_count += 1

            entry_categories = self._extract_categories(entry)
            if not entry_categories:
                logger.debug(f"No category found for entry {entry.get('link')}")

            if not self._category_hit(entry_categories, category_set):
                logger.debug(f"Skipped ePrint {_entry_get(entry, 'link')} with categories: {sorted(entry_categories)}")
                continue
            category_matched_count += 1

            matched_entries.append((entry_time, entry))
            if self.config.executor.debug and len(matched_entries) >= 10:
                break

        matched_dates = ', '.join(f"{d}:{count}" for d, count in sorted(date_hits.items()))
        logger.info(f"ePrint entry date histogram: {matched_dates}")
        logger.info(f"ePrint date match count: {matched_today_count}")
        logger.info(f"ePrint category match count: {category_matched_count}")

        papers = [entry for _, entry in sorted(matched_entries, key=lambda item: item[0], reverse=True)]
        if self.config.executor.debug:
            return papers[:10]
        return papers

    def convert_to_paper(self, raw_paper: Any) -> Paper:
        title = raw_paper.title

        authors = []
        if getattr(raw_paper, "authors", None):
            for item in raw_paper.authors:
                name = item.get("name")
                if name is not None:
                    authors.append(name)
        if not authors and getattr(raw_paper, "author", None):
            authors = [a.strip() for a in re.split(r",| and ", str(raw_paper.author)) if a.strip()]

        abstract = getattr(raw_paper, "summary", "")
        abstract = re.sub(r"<[^>]+>", "", abstract)
        abstract = unescape(abstract)

        source_note = _entry_get(raw_paper, "comment", None)
        if source_note is not None:
            source_note = str(source_note).strip()
            if source_note == "":
                source_note = None

        pdf_url = None
        for link in getattr(raw_paper, "links", []):
            if link.get("type") == "application/pdf":
                pdf_url = link.get("href")
                break

        return Paper(
            source=self.name,
            source_note=source_note,
            title=title,
            authors=authors,
            abstract=abstract.strip(),
            url=raw_paper.link,
            pdf_url=pdf_url,
            full_text=None,
        )
