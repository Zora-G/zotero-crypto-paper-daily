"""Tests for zotero_arxiv_daily.protocol: Paper.generate_tldr, Paper.generate_affiliations."""

import pytest
from types import SimpleNamespace

from tests.canned_responses import make_sample_paper, make_stub_openai_client


@pytest.fixture()
def llm_params():
    return {
        "language": "English",
        "generation_kwargs": {"model": "gpt-4o-mini", "max_tokens": 16384},
    }


# ---------------------------------------------------------------------------
# generate_tldr
# ---------------------------------------------------------------------------


def test_tldr_returns_response(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_tldr(client, llm_params)
    assert "<strong>Problem：</strong>" in result
    assert "<strong>Method：</strong>" in result
    assert "<strong>Cryptography relevance：</strong>" in result
    assert paper.llm_relevance_score == 8.5
    assert "LLM review" in paper.llm_tags
    assert paper.tldr == result


def test_tldr_without_abstract_or_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(abstract="", full_text=None)
    result = paper.generate_tldr(client, llm_params)
    assert "Failed to generate TLDR" in result


def test_tldr_falls_back_to_abstract_on_error(llm_params):
    paper = make_sample_paper()

    # Client whose create() raises
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("API down")))
        )
    )
    result = paper.generate_tldr(broken_client, llm_params)
    assert result == paper.abstract


def test_tldr_uses_configured_sentence_and_length(llm_params):
    captured_prompt = {}

    def create_with_prompt_capture(**kwargs):
        user_msg = kwargs.get("messages", [])[1]["content"]
        captured_prompt["content"] = user_msg
        return make_stub_openai_client().chat.completions.create(**kwargs)

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_with_prompt_capture)
        )
    )
    llm_params["tldr"] = {"max_sentences": 3, "max_words": 160}
    paper = make_sample_paper()
    result = paper.generate_tldr(client, llm_params)

    assert "<strong>Problem：</strong>" in result
    assert "structured paper judgment" in captured_prompt["content"]
    assert "160" in captured_prompt["content"]


def test_tldr_formats_chinese_structured_review(llm_params):
    client = make_stub_openai_client()
    llm_params["language"] = "Chinese"
    paper = make_sample_paper()
    result = paper.generate_tldr(client, llm_params)
    assert "<strong>问题：</strong>" in result
    assert "<strong>方法：</strong>" in result
    assert "<strong>密码相关性：</strong>" in result
    assert "<strong>AI相关性：</strong>" in result
    assert "<strong>为什么值得看：</strong>" in result


def test_tldr_chinese_falls_back_when_llm_returns_english_json(llm_params):
    client = make_stub_openai_client()
    llm_params["language"] = "Chinese"
    paper = make_sample_paper(title="English Title", title_cn="中文标题")
    result = paper.generate_tldr(client, llm_params)
    assert "<strong>问题：</strong>" in result
    assert "Identify whether a paper is relevant" not in result
    assert "English Title" not in result
    assert "中文标题" in result


def test_tldr_malformed_json_uses_clean_fallback(llm_params):
    def create_malformed(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="```json\n{bad json\n```"),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_malformed)
        )
    )
    llm_params["language"] = "Chinese"
    paper = make_sample_paper(title="Fallback Title", title_cn="兜底标题")
    result = paper.generate_tldr(client, llm_params)
    assert "<strong>问题：</strong>" in result
    assert "{bad json" not in result
    assert "Fallback Title" not in result
    assert "兜底标题" in result


def test_tldr_escapes_html_from_llm(llm_params):
    def create_html(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"problem":"<script>alert(1)</script>","method":"Use PAKE","cryptography_relevance":"Direct","ai_relevance":"None","why_worth_reading":"Useful","relevance_score":7,"tags":["x"]}'
                    ),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_html)
        )
    )
    paper = make_sample_paper()
    result = paper.generate_tldr(client, llm_params)
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_tldr_truncates_long_prompt(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text="word " * 10000)
    result = paper.generate_tldr(client, llm_params)
    assert result is not None


# ---------------------------------------------------------------------------
# generate_affiliations
# ---------------------------------------------------------------------------


def test_affiliations_returns_parsed_list(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert isinstance(result, list)
    assert "TsingHua University" in result
    assert "Peking University" in result


def test_affiliations_none_without_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text=None)
    result = paper.generate_affiliations(client, llm_params)
    assert result is None


def test_affiliations_deduplicates(llm_params):
    """The stub returns two distinct affiliations, so no dedup needed.
    But confirm the set() dedup in the code doesn't break anything.
    """
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert len(result) == len(set(result))


def test_affiliations_malformed_llm_output(llm_params):
    """LLM returns affiliations without JSON brackets. Should fall back gracefully."""
    from types import SimpleNamespace

    def create_no_brackets(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="TsingHua University, Peking University"),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_no_brackets)
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    # re.search for [...] will fail -> AttributeError -> caught -> returns None
    assert result is None


def test_affiliations_error_returns_none(llm_params):
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(broken_client, llm_params)
    assert result is None
    assert paper.affiliations is None


def test_title_translation_returns_translated_text(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(title="Sample Paper Title")
    result = paper.generate_title_translation(client, llm_params)
    assert result == "示例论文题目（中文）"
    assert paper.title_cn == "示例论文题目（中文）"


def test_title_translation_fallback_on_error(llm_params):
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        )
    )
    paper = make_sample_paper(title="Original English Title")
    result = paper.generate_title_translation(broken_client, llm_params)
    assert result == "Original English Title"
    assert paper.title_cn == "Original English Title"
