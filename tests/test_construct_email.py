"""Tests for zotero_arxiv_daily.construct_email: render_email, get_stars, get_block_html."""

from zotero_arxiv_daily.construct_email import render_email, get_stars, get_block_html, get_empty_html
from tests.canned_responses import make_sample_paper


def test_render_email_with_papers():
    papers = [make_sample_paper(score=7.5, tldr="A great paper.", affiliations=["MIT"])]
    html = render_email(papers)
    assert "Sample Paper Title" in html
    assert "Source:" in html
    assert "arxiv" in html
    assert "A great paper." in html
    assert "MIT" in html


def test_render_email_empty_list():
    html = render_email([])
    assert "No Papers Today" in html


def test_render_email_author_truncation():
    authors = [f"Author {i}" for i in range(10)]
    paper = make_sample_paper(authors=authors, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Author 0" in html
    assert "Author 1" in html
    assert "Author 2" in html
    assert "..." in html
    assert "Author 8" in html
    assert "Author 9" in html
    # Middle authors should be truncated
    assert "Author 5" not in html


def test_render_email_affiliation_truncation():
    affiliations = [f"Uni {i}" for i in range(8)]
    paper = make_sample_paper(affiliations=affiliations, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Uni 0" in html
    assert "Uni 4" in html
    assert "..." in html
    assert "Uni 7" not in html


def test_render_email_no_affiliations():
    paper = make_sample_paper(affiliations=None, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Unknown Affiliation" in html


def test_render_email_with_feedback_buttons():
    paper = make_sample_paper(score=7.2, tldr="Good", url="https://arxiv.org/abs/2501.00001")
    html = render_email(
        [paper],
        feedback_cfg={
            "enabled": True,
            "endpoint": "https://example.com/feedback",
        },
    )
    assert "推送满意" in html
    assert "不太满意" in html
    assert "action=liked" in html
    assert "action=dislike" in html
    assert "paper_url=https%3A%2F%2Farxiv.org%2Fabs%2F2501.00001" in html


def test_render_email_feedback_endpoint_env_override(monkeypatch):
    monkeypatch.setenv("FEEDBACK_ENDPOINT", "https://override.example/feedback")
    paper = make_sample_paper(score=7.2, tldr="Good", url="https://arxiv.org/abs/2501.00001")
    html = render_email(
        [paper],
        feedback_cfg={
            "enabled": True,
            "endpoint": "https://example.com/feedback",
        },
    )
    assert "https://override.example/feedback?action=liked" in html
    assert "https://example.com/feedback" not in html


def test_get_stars_low_score():
    assert get_stars(5.0) == ""
    assert get_stars(6.0) == ""


def test_get_stars_high_score():
    stars = get_stars(8.0)
    assert stars.count("full-star") == 5


def test_get_stars_mid_score():
    stars = get_stars(7.0)
    assert "star" in stars
    assert stars.count("full-star") + stars.count("half-star") > 0


def test_get_block_html_contains_all_fields():
    html = get_block_html("Title", "eprint", "Auth", "3.5", "Summary", "http://pdf.url", "MIT", "标题")
    assert "Title" in html
    assert "标题" in html
    assert "eprint" in html
    assert "Auth" in html
    assert "3.5" in html
    assert "Summary" in html
    assert "http://pdf.url" in html
    assert "MIT" in html


def test_get_block_html_source_note():
    html = get_block_html("Title", "arxiv", "Auth", "3.5", "Summary", "http://pdf.url", "MIT", "标题", "Nice comments from author")
    assert "Source:</strong> arxiv - Nice comments from author" in html


def test_render_email_shows_chinese_title():
    paper = make_sample_paper(score=7.0, tldr="ok", title_cn="中文题目")
    html = render_email([paper])
    assert "中文题目" in html


def test_get_empty_html():
    html = get_empty_html()
    assert "No Papers Today" in html
