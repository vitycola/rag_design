# rag_design

A hand-rolled RAG (Retrieval-Augmented Generation) pipeline built from first principles as a portfolio demonstration of AI backend design. No LangChain or LlamaIndex — just clean hexagonal architecture with swappable ports and adapters.

## Architecture

The pipeline is organized as 8 independent modules behind ABCs (ports), wired together by a thin `RAGPipeline` orchestrator:

```
config.py / models.py / errors.py   ← shared kernel (no internal imports)
        │
        ├── ingestion/    PDFLoader, MarkdownLoader, URLLoader via LoaderRegistry
        ├── chunking/     SemanticChunker (sentence-transformers) + FixedSizeChunker fallback
        ├── embeddings/   VoyageEmbedder (voyage-3-lite) + LocalEmbedder (all-mpnet-base-v2)
        ├── store/        ChromaVectorStore (persistent, cosine similarity, pre-computed embeddings)
        ├── retrieval/    Retriever (embed query → store.query → threshold filter)
        ├── generation/   ClaudeGenerator (claude-sonnet-4-5, prompt caching, citation parsing)
        ├── pipeline/     RAGPipeline.from_config() — constructor-injected via factories
        └── evaluation/   RAGAS harness (faithfulness, answer_relevancy, context_precision, context_recall)
```

Key design decision: only `factories.py` and `evaluation/run.py` know concrete adapter names. Everything else depends on ports. Swapping Voyage → local embedder requires one env var change, no code edits.

## Setup

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY and VOYAGE_API_KEY
```

To use the local embedder instead of Voyage (no API key required):

```bash
# In .env:
EMBEDDER=local
```

## Usage

```bash
# Ingest a document (PDF, Markdown, or URL)
rag ingest data/raw/book.pdf
rag ingest https://example.com/article

# Query the knowledge base
rag query "What is the difference between semantic and keyword search?"

# Show indexed document count
rag stats
```

## Evaluation

```bash
# Run RAGAS metrics against the golden Q&A set
python -m rag_design.evaluation.run --golden data/golden_set.json --output data/processed/eval.json
```

Thresholds (non-zero exit if any missed):
- faithfulness > 0.90
- answer_relevancy > 0.85
- context_precision > 0.80
- context_recall > 0.75

## License note

This project uses [PyMuPDF](https://pymupdf.readthedocs.io/) (`pymupdf4llm`) which is licensed under **AGPL-3.0**. If you distribute a modified version of this software that uses PyMuPDF, the AGPL terms apply. For commercial use without AGPL obligations, a PyMuPDF commercial license is available from Artifex.

## Portfolio context

This project was built to demonstrate production-grade RAG system design: clean architecture, provider abstraction, prompt caching, semantic chunking, and automated evaluation — without framework magic hiding the fundamentals.
