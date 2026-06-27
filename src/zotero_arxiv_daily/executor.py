from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper
import random
from datetime import datetime
from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email
from .quality import OpenAlexQualityEnricher
from openai import OpenAI
from tqdm import tqdm


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
        reranked_papers = []
        if len(all_papers) > 0:
            logger.info("Reranking papers...")
            reranked_papers = self.reranker.rerank(all_papers, corpus)
            source_minimums = self._source_minimums(self.config)
            if source_minimums:
                logger.info(f"Applying source minimums: {source_minimums}")
            reranked_papers = self._apply_source_minimums(reranked_papers, source_minimums)
            logger.info("Generating title translation, TLDR and affiliations...")
            for p in tqdm(reranked_papers):
                p.generate_title_translation(self.openai_client, self.config.llm)
                p.generate_tldr(self.openai_client, self.config.llm)
                p.generate_affiliations(self.openai_client, self.config.llm)
        elif not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return
        logger.info("Sending email...")
        email_content = render_email(reranked_papers, feedback_cfg=getattr(self.config, "feedback", None))
        send_email(self.config, email_content)
        logger.info("Email sent successfully")
