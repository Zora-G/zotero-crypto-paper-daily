from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
import ast
from html import escape
RawPaperItem = TypeVar('RawPaperItem')


def _truncate_prompt(prompt: str, token_limit: int, model: str = "gpt-4o") -> str:
    try:
        enc = tiktoken.encoding_for_model(model)
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:token_limit]
        return enc.decode(prompt_tokens)
    except Exception as e:
        logger.debug(f"Failed to tokenize prompt with tiktoken: {e}")
        return prompt[: token_limit * 4]

@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    source_note: Optional[str] = None
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    title_cn: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None
    citation_count: Optional[int] = None
    author_h_index: Optional[float] = None
    venue_name: Optional[str] = None
    venue_type: Optional[str] = None
    llm_relevance_score: Optional[float] = None
    llm_tags: Optional[list[str]] = None
    topic_cluster_id: Optional[int] = None
    topic_cluster_size: Optional[int] = None
    related_papers: Optional[list[str]] = None

    @staticmethod
    def _is_chinese_language(lang: str) -> bool:
        return str(lang).strip().lower().startswith(("chinese", "zh", "中文"))

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return bool(re.search(r"[\u3400-\u9fff]", text or ""))

    @staticmethod
    def _extract_json_object(raw: str) -> dict | None:
        if not raw:
            return None

        text = raw.strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
            text = re.sub(r"```$", "", text).strip()

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None

        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

        try:
            parsed = ast.literal_eval(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    @staticmethod
    def _coerce_relevance_score(value) -> Optional[float]:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        if score < 0:
            return 0.0
        if score > 10:
            return 10.0
        return score

    @staticmethod
    def _format_structured_tldr(review: dict, lang: str) -> str:
        is_chinese = Paper._is_chinese_language(lang)
        labels = {
            "problem": "问题" if is_chinese else "Problem",
            "method": "方法" if is_chinese else "Method",
            "cryptography_relevance": "密码相关性" if is_chinese else "Cryptography relevance",
            "ai_relevance": "AI相关性" if is_chinese else "AI relevance",
            "why_worth_reading": "为什么值得看" if is_chinese else "Why worth reading",
        }

        lines = []
        for key, label in labels.items():
            value = str(review.get(key, "") or "").strip()
            if value:
                lines.append(f"<strong>{label}：</strong>{escape(value)}")
        return "<br>".join(lines)

    def _fallback_structured_tldr(self, lang: str) -> str:
        is_chinese = self._is_chinese_language(lang)
        if is_chinese:
            abstract = (self.abstract or "").strip()
            title = (self.title_cn or "").strip()
            if not self._contains_cjk(title):
                title = "该论文"
            if self._contains_cjk(abstract):
                method = abstract[:220] + ("..." if len(abstract) > 220 else "")
            else:
                method = "模型未稳定返回可解析的中文方法说明；建议结合论文原文快速确认具体技术路线。"
            review = {
                "problem": f"围绕“{title}”提出的问题展开。",
                "method": method,
                "cryptography_relevance": "模型本次没有稳定返回可解析的密码学相关性判断；请结合题目、摘要和来源进一步确认。",
                "ai_relevance": "模型本次没有稳定返回可解析的 AI 相关性判断；若题目或摘要涉及模型、数据、隐私保护 AI 或智能体，可优先查看。",
                "why_worth_reading": "该论文通过基础检索和排序进入候选列表，值得用 PDF 快速核对是否符合你的研究兴趣。",
            }
            return self._format_structured_tldr(review, "Chinese")

        review = {
            "problem": f"Studies the problem indicated by the title: {self.title}.",
            "method": (self.abstract or "The abstract is unavailable.")[:220],
            "cryptography_relevance": "The model did not return a stable parseable judgment; inspect the title, abstract, and source.",
            "ai_relevance": "The model did not return a stable parseable AI relevance judgment.",
            "why_worth_reading": "The paper passed retrieval and ranking filters, so it is worth a quick PDF check.",
        }
        return self._format_structured_tldr(review, lang)

    @staticmethod
    def _review_language_mismatch(review: dict, lang: str) -> bool:
        if not Paper._is_chinese_language(lang):
            return False
        values = [
            str(review.get(key, "") or "")
            for key in (
                "problem",
                "method",
                "cryptography_relevance",
                "ai_relevance",
                "why_worth_reading",
            )
        ]
        combined = "".join(values)
        if len(combined.strip()) < 20:
            return False
        return not Paper._contains_cjk(combined)

    def _generate_tldr_with_llm(self, openai_client:OpenAI,llm_params:dict) -> str:
        lang = llm_params.get('language', 'English')
        tldr_cfg = llm_params.get("tldr", {})
        max_sentences = int(tldr_cfg.get("max_sentences", 1))
        if max_sentences < 1:
            max_sentences = 1
        max_words = tldr_cfg.get("max_words", 120)
        if max_words is not None:
            max_words = int(max_words)
            if max_words < 20:
                max_words = 20
            length_hint = f"Keep the summary under around {max_words} words."
        else:
            length_hint = "Keep the summary concise."
        if max_sentences == 1:
            sentence_hint = "generate a one-sentence"
        else:
            sentence_hint = f"generate a {max_sentences}-sentence"

        review_profile = llm_params.get("review_profile") or []
        if isinstance(review_profile, str):
            review_profile = [review_profile]
        review_profile = [str(item).strip() for item in review_profile if str(item).strip()]

        prompt = (
            f"Given the following information of a paper, generate a structured paper judgment in {lang}.\n"
            f"{length_hint}\n"
            "Output must be strict JSON only: no markdown, no code fences, no comments, no extra text.\n"
            "Use double quotes for all JSON keys and string values.\n"
            "Return only one valid JSON object with these keys:\n"
            "- problem: the concrete research problem or new application scenario.\n"
            "- method: the main technique, construction, primitive, protocol, attack, or system idea.\n"
            "- cryptography_relevance: how directly it relates to cryptography, applied cryptography, privacy, authentication, ZK, MPC, FHE, PSI, credentials, signatures, encryption, blockchain cryptography, or security proofs.\n"
            "- ai_relevance: whether and how it relates to AI/ML systems, privacy-preserving AI, model security, data governance, or AI agents; say low/none if it does not.\n"
            "- why_worth_reading: why this paper is worth reading for a researcher interested in applied cryptography, new cryptographic primitives, and privacy-preserving systems.\n"
            "- relevance_score: a number from 0 to 10 for this user's interests.\n"
            "- tags: an array of 3 to 6 concise technical tags.\n"
            f"For each natural-language field, {sentence_hint} explanation is enough.\n"
            "Be specific, technical, and avoid generic praise.\n\n"
        )
        if self._is_chinese_language(lang):
            prompt += (
                "All natural-language field values MUST be written in Simplified Chinese. "
                "Keep English acronyms such as PAKE, ZK, MPC, FHE, PSI, LLM only when necessary.\n\n"
            )
        if review_profile:
            prompt += "User interest profile:\n"
            for item in review_profile:
                prompt += f"- {item}\n"
            prompt += "\n"
        if self.title:
            prompt += f"Title:\n {self.title}\n\n"

        if self.abstract:
            prompt += f"Abstract: {self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"
        
        prompt = _truncate_prompt(prompt, 4000)
        
        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user. Your answer should be in {lang}.",
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get('generation_kwargs', {})
        )
        content = response.choices[0].message.content or ""
        review = self._extract_json_object(content)
        if not review:
            logger.warning(f"LLM returned malformed structured TLDR for {self.url}; using clean fallback.")
            return self._fallback_structured_tldr(lang)
        if self._review_language_mismatch(review, lang):
            logger.warning(f"LLM returned non-Chinese structured TLDR for {self.url}; using clean fallback.")
            return self._fallback_structured_tldr(lang)

        self.llm_relevance_score = self._coerce_relevance_score(review.get("relevance_score"))
        tags = review.get("tags")
        if isinstance(tags, list):
            self.llm_tags = [str(tag).strip() for tag in tags if str(tag).strip()]
        tldr = self._format_structured_tldr(review, lang)
        return tldr or content
    
    def generate_tldr(self, openai_client:OpenAI,llm_params:dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client,llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            tldr = self.abstract
            self.tldr = tldr
            return tldr

    def _generate_affiliations_with_llm(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        if self.full_text is not None:
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]':\n\n{self.full_text}"
            prompt = _truncate_prompt(prompt, 2000)
            affiliations = openai_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from a paper. You should return a python list of affiliations sorted by the author order, like [\"TsingHua University\",\"Peking University\"]. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ],
                **llm_params.get('generation_kwargs', {})
            )
            affiliations = affiliations.choices[0].message.content

            affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
            affiliations = json.loads(affiliations)
            affiliations = list(set(affiliations))
            affiliations = [str(a) for a in affiliations]

            return affiliations

    def _generate_title_translation_with_llm(self, openai_client: OpenAI, llm_params: dict) -> str:
        if not self.title:
            return self.title or ""

        prompt = (
            "Translate the following academic paper title into Simplified Chinese.\n\n"
            f"Title: {self.title}"
        )
        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise translator for scientific paper titles. "
                        "Translate only the title text, output only the translated title."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get("generation_kwargs", {}),
        )
        return (response.choices[0].message.content or "").strip()
    
    def generate_affiliations(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client,llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = None
            return None

    def generate_title_translation(self, openai_client:OpenAI,llm_params:dict) -> str:
        try:
            translated_title = self._generate_title_translation_with_llm(openai_client,llm_params)
            if not translated_title:
                translated_title = self.title
            self.title_cn = translated_title
            return translated_title
        except Exception as e:
            logger.warning(f"Failed to generate title translation of {self.url}: {e}")
            self.title_cn = self.title
            return self.title
@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
