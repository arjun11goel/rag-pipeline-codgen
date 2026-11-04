# RAG Pipeline — PromptForge Week 3

A local Retrieval-Augmented Generation (RAG) application built with Python, Sentence Transformers, ChromaDB, and the Groq API. It answers questions using passages retrieved from a source document rather than relying only on the language model's general knowledge.

The source is a 4,091-word fictional student technology-support handbook. The document is split into overlapping chunks, embedded with the Week 2 `sentence-transformers/all-MiniLM-L6-v2` model, and stored in a persistent ChromaDB collection. At query time, the question is embedded, relevant passages are retrieved, and Groq generates an answer grounded in that context.

> **Source note:** Northstar Learning Institute is fictional. Its handbook is training material for this exercise, not a real institution's policy document.

## Features

- Configurable word-based chunking with overlap (default: 250 words / 50 words).
- Local 384-dimensional embeddings using `all-MiniLM-L6-v2`.
- Persistent ChromaDB storage for chunk text, IDs, embeddings, and metadata.
- Top-k retrieval with cosine-similarity scores.
- Groq answer generation with a context-only instruction and abstention response.
- Configurable retrieval threshold and CLI debug mode.
- Eight evaluation questions: five answerable and three outside the source document's scope.
- Unit tests for chunking, overlap, retrieval helpers, and grounding behavior.

## Pipeline

```text
source_document.txt -> chunk_document.py -> chunks.json
       -> all-MiniLM-L6-v2 -> persistent ChromaDB
       -> question embedding -> top-k retrieval
       -> similarity gate -> grounded Groq prompt
       -> answer with source chunk IDs / abstention
```

## Technology

| Technology | Role |
|---|---|
| Python 3.13 | Application logic and CLI |
| Sentence Transformers | Local text and query embeddings |
| `all-MiniLM-L6-v2` | 384-dimensional embedding model reused from Week 2 |
| ChromaDB | Persistent local vector store |
| Groq Python SDK | Grounded answer generation |
| `python-dotenv` | Environment-based API configuration |
| NumPy / `unittest` | Embedding array handling / automated tests |

The default Groq model is `openai/gpt-oss-20b`. The model can be changed through `GROQ_MODEL`. The project carries forward the Groq SDK and `.env` configuration approach from Week 1 and the embedding model from Week 2.

## Repository structure

```text
rag-pipeline/
├── data/
│   ├── source_document.txt
│   └── evaluation_questions.json
├── chroma_db/.gitkeep
├── experiments/chunk-size-experiment.md
├── tests/
│   ├── test_chunking.py
│   ├── test_pipeline.py
│   └── test_rag_utils.py
├── chunk_document.py
├── build_vector_store.py
├── rag_query.py
├── rag_utils.py
├── rag_config.py
├── evaluate_rag.py
├── compare_grounding.py
├── requirements.txt
├── .env.example
├── .gitignore
├── reflections.md
└── README.md
```

`data/chunks.json` and ChromaDB files are generated locally and excluded from version control. The local `.env` file is also ignored; only `.env.example` belongs in the repository.

## Setup and usage

The project was tested on Windows with Python 3.13 and CPU inference. A Groq API key and internet connection are required for generation. The first embedding-model load also downloads the model.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

The local `.env` contains the API key and runtime settings:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
RAG_MIN_SIMILARITY=0.25
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

The primary commands are:

```powershell
python -m unittest discover -s tests -v
python chunk_document.py
python build_vector_store.py
python rag_query.py
python evaluate_rag.py
```

The default configuration creates 21 chunks from the source document and rebuilds the `northstar_student_support` ChromaDB collection. The CLI supports `/help`, `/debug on`, `/debug off`, `/topk N` (1–8), and `/exit`. A single question can also be submitted with `python rag_query.py --question "How long can I restore a deleted Cloud Drive file?" --debug`.

`RAG_MIN_SIMILARITY` is a heuristic cosine-similarity cutoff, not a confidence probability. A high score does not prove that the retrieved passage contains the answer.

## Evaluation results

The reported automated test run completed with **17 tests passed**. All three chunk-size trials produced answers for the five in-scope questions and the required abstention response for the three out-of-scope questions.

| Chunk size / overlap | Chunks | Mean top similarity for answerable questions | Output categories |
|---|---:|---:|---|
| 100 / 20 | 51 | 0.654 | 5 generated / 3 abstained |
| 250 / 50 | 21 | 0.590 | 5 generated / 3 abstained |
| 400 / 80 | 13 | 0.551 | 5 generated / 3 abstained |

The 250/50 configuration gave the most consistently complete answers in the captured runs. Smaller chunks produced focused matches for some facts; two answers in the large-chunk run stopped mid-sentence in the captured output. These are observations from one run per configuration, not proof that chunk size alone caused the differences. The evaluation script does not automatically grade semantic correctness. Full observations are recorded in [`experiments/chunk-size-experiment.md`](experiments/chunk-size-experiment.md) and [`reflections.md`](reflections.md).

Notably, the unsupported tuition-fee question received a top similarity of 0.587 with 250-word chunks but was still rejected. This shows why similarity is a retrieval-ranking signal rather than an answer-confidence score.

## Grounding and limitations

Two safeguards are used: a retrieval gate rejects low-similarity results, and the generation prompt requires answers to use only the retrieved passages. When the context is insufficient, the required response is `I don't know based on the provided context.` These safeguards reduce unsupported answers but do not guarantee factuality; retrieval can miss evidence and an LLM can misinterpret it.

`compare_grounding.py` contains a separate comparison between a basic context prompt and the explicit grounding prompt. Its measured before/after output should be recorded separately from the main evaluation results. The dataset and evaluation set are small and instructional, not a production benchmark.

## References

- [Sentence Transformers model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Sentence Transformers documentation](https://sbert.net/docs/)
- [Chroma documentation](https://docs.trychroma.com/)
- [Groq API documentation](https://console.groq.com/docs/)
