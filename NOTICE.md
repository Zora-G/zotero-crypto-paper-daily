# Notice

## Upstream

This repository is based on:

- Project: `TideDra/zotero-arxiv-daily`
- URL: https://github.com/TideDra/zotero-arxiv-daily
- License: AGPLv3

The original project recommends new arXiv papers by comparing daily arXiv papers with a user's Zotero library and sending the result by email through GitHub Actions.

## Local Modifications

This repository adds and configures a cryptography-focused daily paper workflow:

- Added IACR ePrint retrieval alongside arXiv retrieval.
- Added configurable multi-day retrieval windows for arXiv and ePrint.
- Added mixed-source output controls, including per-source minimum paper counts.
- Added source labels and source notes in the generated email.
- Added LLM-based Chinese title translation under each English title.
- Added configurable TLDR length for less compressed paper summaries.
- Added explicit interest-profile reranking for applied cryptography, new primitives, new scenarios, and privacy-preserving cryptographic protocols.
- Added optional OpenAlex-based quality signals, including citation count, author h-index approximation, and venue type.
- Added optional email feedback buttons and a minimal feedback collection server.
- Added feedback-history reranking support for future preference adaptation.
- Adjusted default configuration for arXiv `cs.CR`, IACR ePrint, 10-day retrieval, and a 30-paper email limit.
- Added tests for ePrint retrieval, source balancing, feedback-aware reranking, email rendering, title translation, and configuration behavior.

## Attribution

Please keep this notice and the upstream license when redistributing or modifying this project.
