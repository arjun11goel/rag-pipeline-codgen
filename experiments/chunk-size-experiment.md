# Chunk-size experiment results

The same source document and eight evaluation questions were used for all three trials. The ChromaDB collection was rebuilt after each chunking configuration.

## Summary

| Configuration | Chunk count | Word-count range | Mean top similarity (five answerable questions) | Evaluation categories |
|---|---:|---:|---:|---|
| 100 words / 20 overlap | 51 | 91–100 | 0.654 | 5 generated answers / 3 abstentions |
| 250 words / 50 overlap | 21 | 91–250 | 0.590 | 5 generated answers / 3 abstentions |
| 400 words / 80 overlap | 13 | 251–400 | 0.551 | 5 generated answers / 3 abstentions |

Mean similarities are arithmetic means of the top-chunk scores printed by `evaluate_rag.py` for the five answerable questions. They are retrieval-ranking observations, not accuracy or confidence scores.

## Answerable questions

| Question | 100 / 20 | 250 / 50 | 400 / 80 |
|---|---|---|---|
| Campus secure Wi-Fi | `chunk_0008` · 0.588 | `chunk_0004` · 0.587 | `chunk_0003` · 0.536 |
| Technology support ticket | `chunk_0039` · 0.571 | `chunk_0016` · 0.589 | `chunk_0010` · 0.490 |
| Cloud Drive recycle bin | `chunk_0025` · 0.676 | `chunk_0011` · 0.508 | `chunk_0007` · 0.570 |
| Library laptop loan | `chunk_0043` · 0.696 | `chunk_0018` · 0.541 | `chunk_0011` · 0.509 |
| Failed Learning Hub upload | `chunk_0019` · 0.738 | `chunk_0008` · 0.724 | `chunk_0005` · 0.651 |

## Unanswerable questions

| Question | 100 / 20 | 250 / 50 | 400 / 80 | Result in all trials |
|---|---:|---:|---:|---|
| Monthly tuition fee | 0.508 | 0.587 | 0.587 | Abstained |
| Individual student's database grade | 0.425 | 0.364 | 0.382 | Abstained |
| Current prime minister of Japan | 0.051 | 0.025 | 0.013 | Abstained |

The tuition question is notable because its top similarity score was above the configured 0.25 threshold in every trial, but the assistant still abstained because the retrieved context did not contain a tuition amount. The result demonstrates that semantic similarity alone cannot determine whether a question is answerable.

## Observations

- **100 / 20:** The 51 smaller chunks produced focused matches for several direct factual questions. The Cloud Drive and laptop-loan questions had higher top similarities than in the other trials. The support-ticket answer was more condensed than the response observed with 250-word chunks.
- **250 / 50:** The 21-chunk index gave the most consistently complete answers in the captured outputs. The retrieved passages provided enough surrounding context for the support-ticket and assignment-upload responses without making the index as fragmented as the smallest configuration.
- **400 / 80:** The 13-chunk index returned broader passages and lower mean top similarity across the five answerable questions. The captured Wi-Fi and assignment-upload responses ended mid-sentence; the other three in-scope responses were complete. A single run cannot establish whether chunk size caused those incomplete generations.

Across all trials, the evaluation script classified all five in-scope questions as answer-generating and all three out-of-scope questions as abstentions. This is not a semantic accuracy score: the script prints results for manual inspection rather than automatically grading answer completeness or factual correctness.

## Selected configuration

The final project configuration is **250 words with 50 words of overlap**. It provided the best balance in the observed runs between passage focus, context completeness, and index size. The selection is specific to this document and small evaluation set; a larger labelled test set would be needed for a stronger conclusion.
