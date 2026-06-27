"""Tests for BaseReranker: scoring, sorting, time decay, unknown reranker."""

import numpy as np
import pytest
from omegaconf import open_dict
import csv

from zotero_arxiv_daily.reranker.base import BaseReranker, get_reranker_cls
from tests.canned_responses import make_sample_paper, make_sample_corpus


class StubReranker(BaseReranker):
    """Reranker with a controlled similarity matrix for deterministic tests."""

    def __init__(self, sim_matrix: np.ndarray):
        self.config = None
        self._sim = sim_matrix

    def get_similarity_score(self, s1, s2):
        return self._sim


class SequenceStubReranker(BaseReranker):
    """Reranker returning a different controlled matrix for each similarity call."""

    def __init__(self, sim_matrices: list[np.ndarray]):
        self.config = None
        self._sim_matrices = list(sim_matrices)

    def get_similarity_score(self, s1, s2):
        return self._sim_matrices.pop(0)


def test_rerank_scores_and_sorts():
    corpus = make_sample_corpus(3)
    papers = [make_sample_paper(title=f"Paper {i}") for i in range(2)]

    # Paper 1 has higher similarity to all corpus papers
    sim = np.array([
        [0.1, 0.1, 0.1],  # paper 0 — low
        [0.9, 0.9, 0.9],  # paper 1 — high
    ])
    reranker = StubReranker(sim)
    ranked = reranker.rerank(papers, corpus)
    assert ranked[0].title == "Paper 1"
    assert ranked[1].title == "Paper 0"
    assert ranked[0].score > ranked[1].score


def test_rerank_time_decay_weighting():
    corpus = make_sample_corpus(3)
    papers = [make_sample_paper(title="P")]

    # Only similar to the oldest paper (index 2 after reverse-sort by date)
    sim = np.array([[0.0, 0.0, 1.0]])
    reranker = StubReranker(sim)
    ranked_old = reranker.rerank(papers, corpus)
    score_old = ranked_old[0].score

    # Only similar to the newest paper (index 0 after reverse-sort by date)
    papers2 = [make_sample_paper(title="P")]
    sim2 = np.array([[1.0, 0.0, 0.0]])
    reranker2 = StubReranker(sim2)
    ranked_new = reranker2.rerank(papers2, corpus)
    score_new = ranked_new[0].score

    # Newest corpus paper gets higher time-decay weight, so score should be higher
    assert score_new > score_old


def test_rerank_single_candidate_single_corpus():
    corpus = make_sample_corpus(1)
    papers = [make_sample_paper()]
    sim = np.array([[0.5]])
    reranker = StubReranker(sim)
    ranked = reranker.rerank(papers, corpus)
    assert len(ranked) == 1
    assert ranked[0].score is not None


def test_rerank_quality_boost_prefers_high_quality_paper(config):
    corpus = make_sample_corpus(1)
    papers = [
        make_sample_paper(title="Low quality paper"),
        make_sample_paper(title="High quality paper"),
    ]
    papers[0].citation_count = 0
    papers[0].author_h_index = 1
    papers[1].citation_count = 80
    papers[1].author_h_index = 18
    papers[1].venue_type = "conference"

    sim = np.array([[0.5], [0.5]])
    reranker = StubReranker(sim)
    reranker.config = config

    with open_dict(config.executor):
        config.executor.quality_boost = True
        config.executor.citation_weight = 0.6
        config.executor.author_weight = 0.4
        config.executor.venue_weight = 0.2
        config.executor.citation_scale = 200
        config.executor.author_h_index_scale = 25
        config.executor.interest_profile_weight = 0.0

    ranked = reranker.rerank(papers, corpus)
    assert ranked[0].title == "High quality paper"
    assert ranked[0].score > ranked[1].score


def test_rerank_interest_profile_prefers_matching_paper(config):
    corpus = make_sample_corpus(1)
    papers = [
        make_sample_paper(title="Generic cryptanalysis paper"),
        make_sample_paper(title="New applied cryptographic primitive"),
    ]

    corpus_sim = np.array([[0.5], [0.5]])
    profile_sim = np.array([[0.1], [0.9]])
    reranker = SequenceStubReranker([corpus_sim, profile_sim])
    reranker.config = config

    with open_dict(config.executor):
        config.executor.quality_boost = False
        config.executor.interest_profile = ["new applied cryptographic primitives"]
        config.executor.interest_profile_weight = 1.0

    ranked = reranker.rerank(papers, corpus)
    assert ranked[0].title == "New applied cryptographic primitive"
    assert ranked[0].score > ranked[1].score


def test_rerank_feedback_profiles_influence_ranking(tmp_path, config):
    corpus = make_sample_corpus(1)
    papers = [
        make_sample_paper(title="Password hashing with PAKE"),
        make_sample_paper(title="Classical cryptanalysis paper"),
    ]
    corpus_sim = np.array([[0.5], [0.5]])
    profile_like_sim = np.array([[0.8], [0.1]])
    profile_dislike_sim = np.array([[0.05], [0.9]])
    reranker = SequenceStubReranker([corpus_sim, profile_like_sim, profile_dislike_sim])
    reranker.config = config

    history = tmp_path / "feedback.csv"
    with history.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "action", "source", "title", "paper_url"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "2026-06-26T00:00:00",
                "action": "liked",
                "source": "arxiv",
                "title": "PAKE",
                "paper_url": "https://arxiv.org/abs/1",
            }
        )
        writer.writerow(
            {
                "timestamp": "2026-06-26T00:00:00",
                "action": "dislike",
                "source": "arxiv",
                "title": "cryptanalysis",
                "paper_url": "https://arxiv.org/abs/2",
            }
        )

    with open_dict(config.feedback):
        config.feedback.enabled = True
        config.feedback.history_path = str(history)
        config.feedback.history_max_items = 10
        config.feedback.positive_weight = 1.0
        config.feedback.negative_weight = 1.0
    with open_dict(config.executor):
        config.executor.interest_profile_weight = 0.0

    ranked = reranker.rerank(papers, corpus)
    assert ranked[0].title == "Password hashing with PAKE"


def test_get_reranker_cls_unknown():
    with pytest.raises(ValueError, match="not found"):
        get_reranker_cls("nonexistent_reranker_xyz")
