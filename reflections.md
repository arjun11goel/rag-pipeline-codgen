# PromptForge Week 3 — Reflections

## 1. What does grounding mean in RAG, and why can it reduce hallucination?

Grounding means basing an answer on evidence retrieved from the selected document instead of allowing the model to answer from general knowledge. In this project, the question is embedded with `all-MiniLM-L6-v2`, ChromaDB retrieves the closest handbook passages, and those passages are sent to Groq as context. The prompt restricts the answer to that context and requires `I don't know based on the provided context.` when the evidence is insufficient.

This reduces the opportunity for hallucination because the model has source material and a clear boundary for its answer. The retrieval threshold adds another safeguard by rejecting weak matches. Grounding is not a guarantee: retrieval can miss relevant information, a chunk can omit an important condition, and the model can still misinterpret the text. The retrieved passages and answer must be checked together.

## 2. Why does chunk overlap matter?

A long document is easier to search when it is divided into smaller passages, but an explanation or exception can cross a chunk boundary. Without overlap, one chunk may contain the start of a procedure while the next contains a condition that changes its meaning. Retrieving only one side can leave the model with incomplete evidence.

The default is 250 words with a 50-word overlap. Repeating the end of one chunk at the start of the next preserves some continuity. Too much overlap, however, creates near-duplicate chunks and can make top-k results repetitive. The overlap needs to preserve context without making retrieval redundant.

## 3. What happened with smaller and larger chunk sizes?

I tested the same eight questions with three configurations and rebuilt ChromaDB after each change.

| Configuration | Chunks | Mean top similarity across five answerable questions | Evaluation output |
|---|---:|---:|---|
| 100 words / 20 overlap | 51 | 0.654 | 5 generated answers; 3 abstentions |
| 250 words / 50 overlap | 21 | 0.590 | 5 generated answers; 3 abstentions |
| 400 words / 80 overlap | 13 | 0.551 | 5 generated answers; 3 abstentions |

The 100-word chunks gave focused matches for several factual questions. The Cloud Drive retention question retrieved `chunk_0025` at 0.676, and the laptop-loan question retrieved `chunk_0043` at 0.696. The support-ticket answer was more condensed than the response from the 250-word configuration.

The 250-word configuration produced the most consistently complete answers in the captured runs. It retained useful surrounding detail for the support-ticket and assignment-upload questions without creating as many records as the smallest configuration.

The 400-word configuration reduced the index to 13 chunks. The Cloud Drive and laptop-loan answers were complete, but the captured Wi-Fi and assignment-upload answers stopped mid-sentence. This could reflect generation variability or output capture; the logs do not prove that larger chunks caused the truncation.

All three trials generated answers for the five in-scope questions and abstained on the three out-of-scope questions. The smallest chunks had the highest average similarity, but that did not make them the best overall configuration. Similarity is a ranking signal, not an accuracy score. I kept 250 words with 50 words of overlap because it gave the best balance in these observed runs.

## 4. What did ChromaDB provide that Week 2's manual cosine-similarity search did not?

In Week 2, I stored embeddings in a NumPy matrix and calculated cosine similarity against the matrix to rank results. That made the vector comparison process clear, but the application had to manage the vectors and ranking itself.

ChromaDB stores each chunk's ID, embedding, original text, and metadata in a persistent collection. The application submits a query embedding and asks the collection for the nearest records instead of calculating and sorting all similarities itself. Its local `PersistentClient` keeps the collection on disk between runs. ChromaDB simplifies storage and retrieval, while the Week 2 implementation remains useful for understanding the underlying vector calculations. It does not remove the need to design good chunks or evaluate retrieval quality.

## 5. What did the grounding tests show?

Across the three chunk configurations, all five in-scope questions generated answers and all three out-of-scope questions returned the required abstention response.

The tuition-fee question is a useful example. Its top similarity was 0.508 with 100-word chunks and 0.587 with both the 250-word and 400-word configurations, yet the assistant still abstained because the handbook contains no tuition amount. The question about Japan's current prime minister had very low similarity in all three trials (0.051, 0.025, and 0.013) and was also rejected. The retrieval gate filters weak matches; the grounding prompt handles cases where a seemingly related passage still does not contain the requested fact.

The separate weak-prompt versus strict-prompt comparison is implemented in `compare_grounding.py`, but its before/after terminal output was not included in the logs used for this report. I have therefore not claimed a measured baseline response or invented a difference. This is the one additional result needed to complete the task sheet's explicit prompt-comparison requirement.

## 6. What would I improve next?

The evaluation set is small and manually reviewed, so it cannot establish general retrieval accuracy. I would add more paraphrased and boundary-sensitive questions, check whether each expected fact appears in the retrieved context, and repeat the evaluation. I would also tune the similarity threshold against both answerable and unanswerable examples instead of treating 0.25 as a universal cutoff. A reranking stage could be considered if a larger evaluation set shows that retrieval quality needs it.

## Implementation summary

- Source: fictional Northstar technology-support handbook (4,091 words).
- Final chunking: 250 words with 50 words of overlap (21 chunks).
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
- Vector store: persistent ChromaDB collection `northstar_student_support`.
- Generation: Groq API, default model `openai/gpt-oss-20b`.
- Retrieval: top three chunks; displayed similarity is `1 - cosine distance`.
- Default retrieval threshold: `RAG_MIN_SIMILARITY=0.25` (heuristic, not confidence).
- Automated tests: 17 passed in the reported test run.
