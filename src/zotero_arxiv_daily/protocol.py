from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
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

        prompt = (
            f"Given the following information of a paper, {sentence_hint} TLDR summary in {lang}.\n"
            f"{length_hint}\n"
            "The summary should capture the core contribution and the main technical idea, "
            "and avoid fluff.\n\n"
        )
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
        tldr = response.choices[0].message.content
        return tldr
    
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
