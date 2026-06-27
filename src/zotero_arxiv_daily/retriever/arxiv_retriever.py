from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
import re
from queue import Empty
from time import sleep
from datetime import date, datetime, timezone, timedelta
from typing import Any, Callable, TypeVar
from loguru import logger
import requests

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180
ARXIV_LIST_SHOW = 2000


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    @staticmethod
    def _normalize_category(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        return str(getattr(value, "term", value)).strip()

    @staticmethod
    def _days_back(cfg) -> int:
        days_back = int(getattr(cfg.source.arxiv, "days_back", 1))
        if days_back < 1:
            logger.warning(f"Invalid arXiv days_back={days_back}; reset to 1.")
            days_back = 1
        return days_back

    @staticmethod
    def _extract_entry_publish_dt(entry: Any) -> datetime | None:
        parsed_time = getattr(entry, "published_parsed", None)
        if parsed_time is not None:
            try:
                return datetime(*parsed_time[:6], tzinfo=timezone.utc)
            except Exception:
                pass

        raw_time = getattr(entry, "published", None)
        if raw_time is None:
            raw_time = getattr(entry, "updated", None)
        if raw_time is None:
            return None

        for value in (str(raw_time), str(raw_time).replace("Z", "+00:00")):
            try:
                parsed = datetime.fromisoformat(value)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed
            except Exception:
                pass
        for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
            try:
                parsed = datetime.strptime(raw_time, fmt)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed
            except Exception:
                continue
        return None

    @staticmethod
    def _paper_publish_time(paper: ArxivResult) -> datetime:
        raw = getattr(paper, "published", None)
        if raw is None:
            return datetime.min.replace(tzinfo=timezone.utc)
        if isinstance(raw, datetime):
            return raw
        for value in (str(raw), str(raw).replace("Z", "+00:00")):
            try:
                parsed = datetime.fromisoformat(value)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed
            except Exception:
                pass
        try:
            return datetime.fromtimestamp(float(raw))
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    @staticmethod
    def _match_arxiv_category(result: ArxivResult, allowed: set[str], include_cross_list: bool) -> bool:
        primary_category = ArxivRetriever._normalize_category(getattr(result, "primary_category", None))
        category_values = set()
        for c in getattr(result, "categories", []):
            normalized = ArxivRetriever._normalize_category(c)
            if normalized:
                category_values.add(normalized)
        if primary_category:
            category_values.add(primary_category)

        if include_cross_list:
            return bool(category_values & allowed)
        if primary_category is None:
            return False
        return primary_category in allowed

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        client = arxiv.Client(num_retries=10, delay_seconds=10)
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        days_back = self._days_back(self.config)
        target_dates = {_utc_today() - timedelta(days=i) for i in range(days_back)}

        if days_back > 1:
            allowed_categories = {self._normalize_category(c) for c in self.config.source.arxiv.category}
            search_query = " OR ".join([f"cat:{c}" for c in sorted(allowed_categories) if c])
            search = arxiv.Search(
                query=f"({search_query})",
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
                max_results=max(200, len(allowed_categories) * 150 * days_back),
            )
            raw_papers = []
            min_date = min(target_dates)
            for result in client.results(search):
                paper_publish = self._paper_publish_time(result)
                if paper_publish.date() < min_date:
                    break
                if not self._match_arxiv_category(result, allowed_categories, include_cross_list):
                    continue
                if paper_publish.date() in target_dates:
                    raw_papers.append(result)
                if self.config.executor.debug and len(raw_papers) >= 10:
                    break

            logger.info(f"Retrieved {len(raw_papers)} arXiv papers via API search.")
            raw_papers = sorted(
                raw_papers,
                key=self._paper_publish_time,
                reverse=True,
            )
            return raw_papers

        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        all_paper_ids = []
        for entry in feed.entries:
            if entry.get("arxiv_announce_type", "new") not in allowed_announce_types:
                continue
            published_dt = self._extract_entry_publish_dt(entry)
            if published_dt is not None and published_dt.date() not in target_dates:
                continue
            all_paper_ids.append(entry.id.removeprefix("oai:arXiv.org:"))
        if len(all_paper_ids) == 0:
            logger.warning(
                f"arXiv RSS returned no entries for {query}; falling back to arXiv web new listings."
            )
            all_paper_ids = self._retrieve_recent_paper_ids_from_web(include_cross_list, target_dates)

        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        raw_papers = self._fetch_papers_by_ids(client, all_paper_ids)
        return sorted(raw_papers, key=self._paper_publish_time, reverse=True)

    def _fetch_papers_by_ids(self, client: arxiv.Client, all_paper_ids: list[str]) -> list[ArxivResult]:
        # Get full information of each paper from arxiv api
        raw_papers = []
        bar = tqdm(total=len(all_paper_ids))
        max_batch_retries = 5
        batch_retry_delay = 30
        for i in range(0, len(all_paper_ids), 20):
            search = arxiv.Search(id_list=all_paper_ids[i:i + 20])
            for attempt in range(max_batch_retries):
                try:
                    batch = list(client.results(search))
                    bar.update(len(batch))
                    raw_papers.extend(batch)
                    break
                except arxiv.HTTPError as exc:
                    if exc.status == 429 and attempt < max_batch_retries - 1:
                        wait = batch_retry_delay * (attempt + 1)
                        logger.warning(f"arXiv API 429 on batch {i // 20}, retry {attempt + 1}/{max_batch_retries} in {wait}s")
                        sleep(wait)
                    else:
                        raise
            if i + 20 < len(all_paper_ids):
                sleep(3)
        bar.close()

        return raw_papers

    def _retrieve_recent_paper_ids_from_web(
        self,
        include_cross_list: bool,
        target_dates: set[date],
    ) -> list[str]:
        categories = list(self.config.source.arxiv.category)
        paper_ids = []
        seen_ids = set()
        for category in categories:
            response = requests.get(
                f"https://arxiv.org/list/{category}/new?show={ARXIV_LIST_SHOW}",
                timeout=DOWNLOAD_TIMEOUT,
            )
            response.raise_for_status()
            for target_date in sorted(target_dates, reverse=True):
                for paper_id in _extract_new_listing_ids(response.text, target_date, include_cross_list):
                    if paper_id in seen_ids:
                        continue
                    seen_ids.add(paper_id)
                    paper_ids.append(paper_id)

        logger.info(f"Retrieved {len(paper_ids)} arXiv paper ids from web fallback for {sorted(target_dates)}")
        return paper_ids

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = extract_text_from_tar(raw_paper)
        source_note = getattr(raw_paper, "comment", None)
        if source_note is not None:
            source_note = str(source_note).strip()
            if source_note == "":
                source_note = None
        if full_text is None:
            full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)
        return Paper(
            source=self.name,
            source_note=source_note,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def _extract_new_listing_ids(
    html: str,
    target_date: date,
    include_cross_list: bool,
) -> list[str]:
    date_match = re.search(r"Showing new listings for [A-Za-z]+, (\d{1,2} [A-Za-z]+ \d{4})", html)
    if date_match is None:
        return []
    announcement_date = datetime.strptime(date_match.group(1), "%d %B %Y").date()
    if announcement_date != target_date:
        return []

    paper_ids = []
    seen_ids = set()
    in_wanted_section = False
    for line in html.splitlines():
        heading_match = re.search(r"<h3>(.*?)</h3>", line)
        if heading_match is not None:
            heading = heading_match.group(1)
            in_wanted_section = heading.startswith("New submissions") or (
                include_cross_list and heading.startswith("Cross submissions")
            )
            continue

        if not in_wanted_section:
            continue

        for paper_id in re.findall(r'href\s*=\s*"/abs/([0-9]+\.[0-9]+(?:v\d+)?)"\s+title="Abstract"', line):
            if paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)
            paper_ids.append(paper_id)
    return paper_ids


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
