from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig, OmegaConf
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper, Paper
import random
from datetime import datetime
from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email
from .quality import OpenAlexQualityEnricher
from openai import OpenAI
from tqdm import tqdm
from difflib import SequenceMatcher
import re


def _cfg_get(cfg, key: str, default=None):
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.zotero.{config_key} must contain only glob pattern strings.")

    return list(patterns)


class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.include_path_patterns = normalize_path_patterns(config.zotero.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, "ignore_path")
        logger.info(f"Configured sources: {list(config.executor.source)}")
        logger.info(f"Configured ePrint categories: {getattr(config.source.eprint, 'category', None)}")
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.quality_enricher = OpenAlexQualityEnricher(config)
        self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)
    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = zot.everything(zot.collections())
        collections = {c['key']:c for c in collections}
        corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
        corpus = [c for c in corpus if c['data']['abstractNote'] != '']
        def get_collection_path(col_key:str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']
        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = paths
        logger.info(f"Fetched {len(corpus)} zotero papers")
        return [CorpusPaper(
            title=c['data']['title'],
            abstract=c['data']['abstractNote'],
            added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
            paths=c['paths']
        ) for c in corpus]
    
    def filter_corpus(self, corpus:list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"Selecting zotero papers matching include_path: {self.include_path_patterns}")
            corpus = [
                c for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]
        if self.ignore_path_patterns:
            logger.info(f"Excluding zotero papers matching ignore_path: {self.ignore_path_patterns}")
            corpus = [
                c for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]
        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus)))
            samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
            logger.info(f"Selected {len(corpus)} zotero papers:\n{samples}\n...")
        return corpus

    

    @staticmethod
    def _source_minimums(config):
        """Read and normalize executor.source_min_papers.

        Returns dict[str, int] with non-negative quotas.
        """
        source_min = getattr(config.executor, "source_min_papers", None)
        if not source_min:
            return {}

        mins = {}
        for source, value in source_min.items():
            try:
                min_count = int(value)
            except (TypeError, ValueError):
                logger.warning(f"Invalid source_min_papers value for {source}: {value}")
                continue

            if min_count < 0:
                logger.warning(f"Invalid source_min_papers value for {source}: {min_count}")
                continue

            mins[str(source)] = min_count

        return mins

    def _apply_source_minimums(self, papers:list, mins:dict[str, int]) -> list:
        """Keep at least configured minimum papers from each source after reranking."""
        if len(papers) == 0 or not mins:
            return papers[:self.config.executor.max_paper_num]

        max_paper_num = int(self.config.executor.max_paper_num)
        by_source = {source: [] for source in mins}
        for paper in papers:
            source = getattr(paper, "source", None)
            if source in by_source:
                by_source[source].append(paper)

        total_minimum = sum(mins.values())
        if total_minimum > max_paper_num:
            logger.warning(
                f"Configured source_min_papers total ({total_minimum}) is larger than max_paper_num ({max_paper_num}). "
                "Use global ranking instead."
            )
            return papers[:max_paper_num]

        selected = []
        selected_ids = set()

        for source, min_count in mins.items():
            quota = min(min_count, len(by_source.get(source, [])))
            for paper in by_source.get(source, [])[:quota]:
                selected.append(paper)
                selected_ids.add(id(paper))

        for paper in papers:
            if len(selected) >= max_paper_num:
                break
            if id(paper) not in selected_ids:
                selected.append(paper)
                selected_ids.add(id(paper))

        return selected[:max_paper_num]

    def _executor_section(self, key: str, default=None):
        section = getattr(self.config.executor, key, None)
        return section if section is not None else default

    @staticmethod
    def _normalize_title(title: str) -> str:
        title = (title or "").lower()
        title = re.sub(r"\barxiv:\d{4}\.\d+(v\d+)?\b", " ", title)
        title = re.sub(r"\b(eprint|iacr)\b", " ", title)
        title = re.sub(r"[^a-z0-9]+", " ", title)
        return re.sub(r"\s+", " ", title).strip()

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]{3,}", (text or "").lower())
            if token not in {"the", "and", "for", "with", "from", "this", "that", "paper"}
        }

    @classmethod
    def _title_similarity(cls, left: Paper, right: Paper) -> float:
        l_title = cls._normalize_title(left.title)
        r_title = cls._normalize_title(right.title)
        if not l_title or not r_title:
            return 0.0

        seq_score = SequenceMatcher(None, l_title, r_title).ratio()
        l_tokens = cls._tokens(l_title)
        r_tokens = cls._tokens(r_title)
        token_score = 0.0
        if l_tokens and r_tokens:
            token_score = len(l_tokens & r_tokens) / len(l_tokens | r_tokens)
        return max(seq_score, token_score)

    @classmethod
    def _topic_similarity(cls, left: Paper, right: Paper) -> float:
        title_score = cls._title_similarity(left, right)
        left_text = f"{left.title} {left.abstract}"
        right_text = f"{right.title} {right.abstract}"
        l_tokens = cls._tokens(left_text)
        r_tokens = cls._tokens(right_text)
        abstract_score = 0.0
        if l_tokens and r_tokens:
            abstract_score = len(l_tokens & r_tokens) / len(l_tokens | r_tokens)
        return max(title_score, abstract_score)

    @staticmethod
    def _append_source_note(paper: Paper, note: str) -> None:
        if not note:
            return
        existing = (paper.source_note or "").strip()
        if existing and note not in existing:
            paper.source_note = f"{existing}; {note}"
        elif not existing:
            paper.source_note = note

    @staticmethod
    def _paper_rank_key(paper: Paper, preferred_sources: list[str]) -> tuple:
        try:
            source_rank = preferred_sources.index(str(paper.source))
        except ValueError:
            source_rank = len(preferred_sources)
        return (
            -source_rank,
            1 if paper.full_text else 0,
            1 if paper.pdf_url else 0,
            float(getattr(paper, "author_h_index", None) or 0.0),
            float(getattr(paper, "citation_count", None) or 0.0),
        )

    def _deduplicate_papers(self, papers: list[Paper]) -> list[Paper]:
        dedup_cfg = self._executor_section("dedup", {})
        if dedup_cfg and not bool(_cfg_get(dedup_cfg, "enabled", True)):
            return papers

        threshold = float(_cfg_get(dedup_cfg, "title_similarity", 0.94) or 0.94)
        preferred_sources = list(_cfg_get(dedup_cfg, "preferred_sources", ["eprint", "arxiv", "biorxiv", "medrxiv"]) or [])

        unique: list[Paper] = []
        duplicate_count = 0
        for paper in papers:
            match_index = None
            for index, existing in enumerate(unique):
                if self._title_similarity(existing, paper) >= threshold:
                    match_index = index
                    break

            if match_index is None:
                unique.append(paper)
                continue

            duplicate_count += 1
            existing = unique[match_index]
            keep_new = self._paper_rank_key(paper, preferred_sources) > self._paper_rank_key(existing, preferred_sources)
            kept = paper if keep_new else existing
            other = existing if keep_new else paper
            if keep_new:
                unique[match_index] = kept

            note = f"Also available from {other.source}: {other.url}"
            self._append_source_note(kept, note)
            kept.related_papers = list(kept.related_papers or [])
            kept.related_papers.append(f"{other.source}: {other.title}")

        if duplicate_count:
            logger.info(f"Deduplicated {duplicate_count} near-identical paper versions.")
        return unique

    def _assign_topic_clusters(self, papers: list[Paper]) -> None:
        cluster_cfg = self._executor_section("topic_cluster", {})
        threshold = float(_cfg_get(cluster_cfg, "topic_similarity", 0.74) or 0.74)
        clusters: list[list[Paper]] = []

        for paper in papers:
            target = None
            for index, cluster in enumerate(clusters):
                if any(self._topic_similarity(paper, existing) >= threshold for existing in cluster):
                    target = index
                    break
            if target is None:
                clusters.append([paper])
            else:
                clusters[target].append(paper)

        for index, cluster in enumerate(clusters, start=1):
            for paper in cluster:
                paper.topic_cluster_id = index
                paper.topic_cluster_size = len(cluster)
                if len(cluster) > 1:
                    self._append_source_note(paper, f"Topic cluster {index}, {len(cluster)} related candidates")

    def _apply_topic_cluster_diversity(self, papers: list[Paper]) -> list[Paper]:
        cluster_cfg = self._executor_section("topic_cluster", {})
        if cluster_cfg and not bool(_cfg_get(cluster_cfg, "enabled", True)):
            return papers

        max_per_cluster = int(_cfg_get(cluster_cfg, "max_per_cluster", 2) or 2)
        if max_per_cluster <= 0 or len(papers) <= 1:
            return papers

        self._assign_topic_clusters(papers)
        selected = []
        skipped = []
        cluster_counts: dict[int, int] = {}
        for paper in papers:
            cluster_id = paper.topic_cluster_id or id(paper)
            count = cluster_counts.get(cluster_id, 0)
            if count < max_per_cluster:
                selected.append(paper)
                cluster_counts[cluster_id] = count + 1
            else:
                skipped.append(paper)

        if skipped:
            logger.info(f"Moved {len(skipped)} papers behind stronger same-topic candidates.")
        return selected + skipped

    def _llm_params_for_review(self):
        try:
            params = OmegaConf.to_container(self.config.llm, resolve=True)
        except Exception:
            params = dict(self.config.llm)
        if not isinstance(params, dict):
            params = {}

        profile = getattr(self.config.executor, "interest_profile", None)
        if profile:
            params["review_profile"] = list(profile) if not isinstance(profile, str) else [profile]
        return params

    def _llm_review_score_weight(self) -> float:
        return float(getattr(self.config.executor, "llm_review_score_weight", 0.4) or 0.0)

    def _apply_llm_review_score(self, paper: Paper) -> None:
        if paper.llm_relevance_score is None:
            return

        weight = self._llm_review_score_weight()
        if weight <= 0:
            return

        base_score = float(paper.score or 0.0)
        paper.score = base_score + (paper.llm_relevance_score - 5.0) * weight

    def run(self):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.error(f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}")
            return
        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            try:
                papers = retriever.retrieve_papers()
            except Exception as exc:
                logger.warning(f"Failed to retrieve from {source}: {exc}")
                continue
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")
        if self.quality_enricher.enabled and len(all_papers) > 0:
            logger.info("Enriching papers with OpenAlex citation and author signals...")
            self.quality_enricher.enrich_papers(all_papers)
        all_papers = self._deduplicate_papers(all_papers)
        reranked_papers = []
        if len(all_papers) > 0:
            logger.info("Reranking papers...")
            reranked_papers = self.reranker.rerank(all_papers, corpus)
            reranked_papers = self._apply_topic_cluster_diversity(reranked_papers)
            source_minimums = self._source_minimums(self.config)
            if source_minimums:
                logger.info(f"Applying source minimums: {source_minimums}")
            reranked_papers = self._apply_source_minimums(reranked_papers, source_minimums)
            logger.info("Generating title translation, TLDR and affiliations...")
            llm_params = self._llm_params_for_review()
            for p in tqdm(reranked_papers):
                p.generate_title_translation(self.openai_client, llm_params)
                p.generate_tldr(self.openai_client, llm_params)
                self._apply_llm_review_score(p)
                p.generate_affiliations(self.openai_client, llm_params)
            reranked_papers = sorted(reranked_papers, key=lambda paper: paper.score or 0.0, reverse=True)
        elif not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return
        logger.info("Sending email...")
        email_content = render_email(reranked_papers, feedback_cfg=getattr(self.config, "feedback", None))
        send_email(self.config, email_content)
        logger.info("Email sent successfully")
