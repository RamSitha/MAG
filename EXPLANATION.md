# MAG: A Metadata-Augmented Retrieval Framework for Efficient Handling of Complex Queries

**Authors:** Ramesh K Bhukya, Ayush Shukla, Prabhat Chandra Shrivastava, Sandeep Sharma

## Purpose of This Notebook

This notebook implements and evaluates **MAG**, a metadata-augmented retrieval framework designed to improve passage retrieval for complex natural-language queries. The central idea is to avoid relying only on raw passage text. Instead, each passage is converted into a compact metadata representation composed of statistically important terms, named entities, noun concepts, topic terms, and semantic keyphrases. A sparse retriever, BM25, is then applied to this enriched metadata corpus.

The notebook is intended as an experimental companion for the IEEE Transactions on Multimedia submission. It contains:

- Dataset preparation for MS MARCO, BEIR SciFact, and SQuAD.
- Baseline retrieval systems using TF-IDF, FAISS dense retrieval, and BM25.
- Ablation variants using YAKE, KeyBERT, Q-RATE-style metadata, and combinations of these methods.
- The final MAG pipeline, which fuses Q-RATE, KeyBERT, and YAKE metadata into a score-weighted BM25 index.
- Top-1 retrieval evaluation using accuracy, precision, recall, F1 score, and average retrieval time.

## High-Level Contribution

MAG treats metadata as an intermediate retrieval representation. Instead of indexing full passages directly, the method extracts query-relevant descriptors from each passage and indexes those descriptors. This gives the retrieval model a more focused representation of passage content and reduces noise from long contexts.

The final MAG implementation uses three complementary metadata sources:

1. **Q-RATE-style linguistic metadata**
   Extracts entities, noun concepts, and latent topic terms.

2. **KeyBERT semantic keyphrases**
   Extracts embedding-based keyphrases using `all-MiniLM-L6-v2`.

3. **YAKE statistical keywords**
   Extracts unsupervised local-statistical keywords and phrases.

The extracted metadata are fused into a weighted keyword corpus. Keywords with larger fused scores are repeated more times in the BM25 document representation, allowing BM25 to treat stronger metadata terms as more important during ranking.

## Notebook Organization

The notebook is organized into three dataset-specific experimental blocks.

| Notebook section | Dataset | Purpose |
|---|---:|---|
| MS MARCO block | 1,000 selected relevant passages | Main web-query passage retrieval experiment |
| BEIR block | SciFact test split with distractor corpus | Scientific claim/document retrieval experiment |
| SQuAD block | First five SQuAD v1.1 articles | Question-to-context passage retrieval experiment |

For each dataset, the notebook evaluates a comparable set of retrieval methods:

- TF-IDF with cosine similarity.
- FAISS using SentenceTransformer embeddings.
- BM25 over preprocessed passage text.
- BM25 over YAKE metadata.
- BM25 over KeyBERT metadata.
- BM25 over combined YAKE and KeyBERT metadata.
- BM25 over Q-RATE metadata.
- MAG variants that combine Q-RATE with YAKE and/or KeyBERT.

The final and most refined version appears in the last code cell under the **Upgraded MAG** section.

## Data Preparation

### MS MARCO

The notebook downloads `microsoft/ms_marco` version `v1.1` from Hugging Face and uses the validation split. For each query, the code keeps passages marked as relevant by the dataset field `is_selected == 1`.

Two JSON files are generated:

- `MSMARCO_verify.json`: maps passage IDs to passage text.
- `MSMARCO_test.json`: maps each passage ID to the list of queries for which that passage is relevant.

The notebook limits the MS MARCO corpus to 1,000 unique relevant passages to keep runtime and memory usage manageable in a Colab-style execution environment.

### BEIR SciFact

The notebook downloads the BEIR `scifact` dataset and loads the test split. The relevance judgments are inverted from the standard query-to-document format into the notebook's document-to-query format.

Two JSON files are generated:

- `BEIR_verify.json`: maps document IDs to the concatenated title and document text.
- `BEIR_test.json`: maps each relevant document ID to the queries/claims for which it is relevant.

The BEIR verification corpus is capped at 2,000 documents. All ground-truth relevant documents are inserted first, and the remaining slots are filled with distractor documents from the corpus.

### SQuAD

The notebook uses `train-v1.1.json` from SQuAD v1.1. It extracts the first five articles and converts them into the same internal retrieval format:

- `SQuAD_verify.json`: maps paragraph IDs to paragraph contexts.
- `SQuAD_test.json`: maps paragraph IDs to the questions associated with each paragraph.

This conversion frames SQuAD as a passage retrieval task: given a question, the retriever must identify the paragraph containing the answer.

## Baseline Methods

### TF-IDF

The TF-IDF baseline performs standard lexical retrieval:

1. Lowercase and tokenize text.
2. Remove stopwords.
3. Stem tokens using Porter stemming.
4. Build a `TfidfVectorizer` representation for all passages.
5. Transform each query into the same TF-IDF space.
6. Retrieve the passage with the highest cosine similarity.

This baseline measures classic bag-of-words lexical matching.

### FAISS Dense Retrieval

The dense baseline uses `sentence-transformers/all-MiniLM-L6-v2` to embed passages and queries. Passage embeddings are indexed with FAISS using `IndexFlatL2`.

Retrieval follows this process:

1. Encode every passage into a dense vector.
2. Add all passage vectors to the FAISS index.
3. Encode the query into the same embedding space.
4. Retrieve the nearest passage by L2 distance.

This baseline evaluates semantic matching through dense vector similarity.

### BM25

The BM25 baseline preprocesses passages and queries using tokenization, stopword removal, and stemming. It then indexes the tokenized passage corpus with `rank_bm25.BM25Okapi`.

This baseline is important because MAG ultimately preserves BM25 as the retrieval engine while changing the indexed representation from raw passage text to metadata-enriched passage descriptors.

## Metadata Extraction Components

### Q-RATE-Style Metadata

The notebook implements a Q-RATE-style feature extraction module using three sources:

| Component | Implementation | Role |
|---|---|---|
| Named Entity Recognition | spaCy `en_core_web_sm` | Captures entity mentions such as people, organizations, dates, locations, and domain terms |
| Topic modeling | Gensim LDA | Identifies latent topic terms from the passage |
| Noun extraction | NLTK POS tagging or spaCy POS tagging | Captures object/concept-bearing terms |

In the final corrected MAG implementation, the noun extractor is revised to use spaCy POS tags and normalized noun frequency. This avoids unstable TF-IDF behavior on a single paragraph and provides a cleaner local importance signal.

### KeyBERT

KeyBERT extracts semantic keyphrases using the lightweight `all-MiniLM-L6-v2` model. In the final MAG implementation, it extracts phrase candidates with one- and two-word n-grams:

```python
keyphrase_ngram_range=(1, 2)
top_n=50
```

KeyBERT contributes embedding-based semantic descriptors that can capture meaning even when exact query wording differs from passage wording.

### YAKE

YAKE provides unsupervised statistical keyword extraction. The implementation uses:

```python
lan="en", n=3, dedupLim=0.9, top=50
```

YAKE returns lower scores for stronger keywords. The final MAG implementation converts YAKE scores into positive importance weights using:

```python
1.0 / (score + 1e-9)
```

This makes highly ranked YAKE keywords contribute more strongly to the fused metadata representation.

## Final MAG Pipeline

The final MAG cell is the most polished implementation in the notebook. Its main steps are:

1. **Install dependencies**
   Required packages include `rank_bm25`, `spacy`, `gensim`, `scikit-learn`, `keybert`, and `yake`.

2. **Load language resources**
   The code downloads required NLTK resources and loads spaCy `en_core_web_sm`.

3. **Define one strict preprocessing pipeline**
   The final implementation uses a single preprocessing function for LDA input, BM25 indexing, and query processing:

```python
def strict_pipeline(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    tokens = word_tokenize(text)
    return [stemmer.stem(word) for word in tokens if word not in stop_words_set]
```

This is important because retrieval quality can degrade when document preprocessing and query preprocessing are inconsistent.

4. **Extract metadata from each passage**
   For each passage, MAG extracts:

- NER scores from spaCy entities.
- LDA topic-term scores.
- Noun frequency scores.
- KeyBERT keyphrase scores.
- YAKE keyword scores.

5. **Fuse metadata scores**
   All metadata sources are accumulated in a `defaultdict(float)`. When the same term appears across multiple metadata generators, its scores are added.

6. **Convert fused scores into a BM25-compatible representation**
   Since BM25 works over token frequencies, MAG converts continuous metadata scores into repeated keyword occurrences. The final code uses:

```python
repeat_count = int(np.ceil((score / max_score) * 4)) + 1
```

Thus, stronger metadata terms appear more often in the indexed representation.

7. **Build the BM25 index**
   The score-weighted metadata documents are preprocessed by `strict_pipeline` and indexed with `BM25Okapi`.

8. **Retrieve**
   Each query is processed using the same strict pipeline. BM25 scores all metadata documents, and the best scoring document is returned.

9. **Apply thresholding**
   The notebook uses a simple query-length-aware threshold:

```python
threshold = max(0.02, min(0.1, query_len * 0.01))
```

If the top BM25 score is lower than this threshold, the result is treated as `No relevant match`.

## Evaluation Protocol

The evaluation is a top-1 retrieval task. For every query, the system retrieves one passage/document. A retrieval is counted as correct when:

```python
retrieved_para_id in test_data and query in test_data[retrieved_para_id]
```

The notebook computes:

- **Accuracy**: fraction of queries for which the top-1 retrieved passage is the annotated relevant passage.
- **Precision**: true positives divided by true positives plus false positives.
- **Recall**: true positives divided by true positives plus false negatives.
- **F1 score**: harmonic mean of precision and recall.
- **Average retrieval time**: mean per-query search time, excluding most offline metadata extraction/index construction costs.

For many top-1 closed-set baselines, precision, recall, F1, and accuracy are numerically identical because each query always returns exactly one passage and the evaluation checks exact passage correctness. In the MAG cells, thresholding can introduce false negatives; therefore precision and recall are computed from explicit true-positive, false-positive, and false-negative counters.

## Reported Results from Notebook Outputs

### MS MARCO

| Method | Queries | Accuracy | Precision | Recall | F1 | Avg. retrieval time |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | 1000 | 0.7840 | 0.7840 | 0.7840 | 0.7840 | 0.003664 s |
| FAISS | 1000 | 0.8770 | 0.8770 | 0.8770 | 0.8770 | 0.017542 s |
| BM25 | 1000 | 0.8300 | 0.8300 | 0.8300 | 0.8300 | 0.001794 s |
| BM25 + YAKE | 1000 | 0.4870 | 0.4870 | 0.4870 | 0.4870 | 0.001111 s |
| BM25 + KeyBERT | 1000 | 0.5540 | 0.5540 | 0.5540 | 0.5540 | 0.001613 s |
| BM25 + Q-RATE | 1000 | 0.7900 | 0.7900 | 1.0000 | 0.8827 | 0.001577 s |
| BM25 + YAKE + KeyBERT | 1000 | 0.5510 | 0.5510 | 0.5510 | 0.5510 | 0.001054 s |
| BM25 + Q-RATE + YAKE | 1000 | 0.8460 | 0.8460 | 1.0000 | 0.9166 | 0.001324 s |
| BM25 + Q-RATE + KeyBERT | 1000 | 0.8610 | 0.8610 | 1.0000 | 0.9253 | 0.001275 s |
| MAG | 1000 | 0.8510 | 0.8510 | 1.0000 | 0.9195 | 0.001169 s |

On MS MARCO, dense retrieval with FAISS gives the strongest accuracy among the reported runs, while MAG-style metadata methods provide competitive accuracy with substantially lower per-query retrieval time than FAISS. The Q-RATE + KeyBERT variant is the strongest metadata-augmented configuration in this block.

### BEIR SciFact

| Method | Queries | Accuracy | Precision | Recall | F1 | Avg. retrieval time |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | 339 | 0.4425 | 0.4425 | 0.4425 | 0.4425 | 0.004960 s |
| FAISS | 339 | 0.5398 | 0.5398 | 0.5398 | 0.5398 | 0.024460 s |
| BM25 | 339 | 0.5664 | 0.5664 | 0.5664 | 0.5664 | 0.006820 s |
| BM25 + YAKE | 339 | 0.1386 | 0.1386 | 0.1386 | 0.1386 | 0.005859 s |
| BM25 + KeyBERT | 339 | 0.2537 | 0.2537 | 0.2537 | 0.2537 | 0.005460 s |
| BM25 + YAKE + KeyBERT | 339 | 0.2714 | 0.2714 | 0.2714 | 0.2714 | 0.005282 s |
| BM25 + Q-RATE | 339 | 0.5251 | 0.5251 | 1.0000 | 0.6886 | 0.007164 s |
| BM25 + Q-RATE + YAKE | 339 | 0.5900 | 0.5900 | 1.0000 | 0.7421 | 0.008896 s |
| BM25 + Q-RATE + KeyBERT | 339 | 0.6047 | 0.6047 | 1.0000 | 0.7537 | 0.010464 s |
| MAG | 339 | 0.6018 | 0.6018 | 1.0000 | 0.7514 | 0.007954 s |

On BEIR SciFact, metadata fusion improves over the raw BM25 baseline in the strongest reported configurations. The Q-RATE + KeyBERT and MAG variants perform best among the reported BEIR runs, indicating that entity/topic/noun metadata is useful for scientific claim retrieval when combined with semantic keyphrases.

### SQuAD

| Method | Queries | Accuracy | Precision | Recall | F1 | Avg. retrieval time |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | 1483 | 0.6339 | 0.6339 | 0.6339 | 0.6339 | 0.002004 s |
| FAISS | 1483 | 0.6399 | 0.6399 | 0.6399 | 0.6399 | 0.020873 s |
| BM25 | 1483 | 0.7141 | 0.7141 | 0.7141 | 0.7141 | 0.002591 s |
| BM25 + YAKE | 1483 | 0.2792 | 0.2792 | 0.2792 | 0.2792 | 0.001046 s |
| BM25 + Q-RATE | 1483 | 0.6979 | 0.6979 | 1.0000 | 0.8221 | 0.000487 s |
| MAG | 1483 | 0.7215 | 0.7215 | 1.0000 | 0.8382 | 0.000489 s |
| Upgraded MAG | 1483 | 0.7310 | 0.7310 | 1.0000 | 0.8446 | 0.000461 s |
| Final corrected MAG | 1483 | 0.7438 | 0.7438 | 1.0000 | 0.8531 | 0.000658 s |

On SQuAD, the final corrected MAG implementation improves over the BM25 baseline in accuracy and F1 while maintaining sub-millisecond average retrieval time in the recorded notebook output.

## Interpretation of Results

The results support three main observations.

First, metadata alone is not sufficient when generated by a single extractor. YAKE-only and KeyBERT-only BM25 variants are often weaker than raw BM25 because they discard too much passage context. This is especially visible in MS MARCO, BEIR, and SQuAD.

Second, Q-RATE-style linguistic metadata is consistently stronger than YAKE-only or KeyBERT-only indexing. Named entities, noun concepts, and topic terms preserve important semantic anchors that are directly useful for passage retrieval.

Third, fusion is the key design choice in MAG. Combining linguistic metadata with semantic/statistical keyphrases produces a richer and more stable index than any single metadata source. The strongest reported metadata variants generally include Q-RATE plus KeyBERT and/or YAKE.

## Reproducibility Notes

The notebook is written for an interactive notebook or Google Colab-style environment. It installs dependencies inside notebook cells using `pip` and downloads external datasets/models during execution.

Important dependencies include:

- `datasets`
- `beir`
- `rank_bm25`
- `sentence-transformers`
- `faiss-cpu`
- `nltk`
- `spacy`
- `gensim`
- `scikit-learn`
- `keybert`
- `yake`

The spaCy model is installed through:

```bash
python -m spacy download en_core_web_sm
```

The notebook downloads or expects the following dataset resources:

- Hugging Face `microsoft/ms_marco`, configuration `v1.1`.
- BEIR `scifact`.
- SQuAD v1.1 `train-v1.1.json`.

Generated intermediate JSON files include:

- `MSMARCO_test.json`
- `MSMARCO_verify.json`
- `BEIR_test.json`
- `BEIR_verify.json`
- `SQuAD_test.json`
- `SQuAD_verify.json`

## Reviewer-Facing Limitations and Clarifications

The current notebook is primarily an experimental prototype. For camera-ready artifact release, the following points should be made explicit or cleaned:

- The reported retrieval times measure online retrieval only. Offline costs such as keyword extraction, embedding generation, model loading, and index construction are not included in the average retrieval time.
- Dataset subsets are capped for practical runtime. MS MARCO uses 1,000 passages, BEIR uses a 2,000-document verification corpus, and SQuAD uses the first five articles.
- The evaluation uses top-1 exact passage matching. It does not report ranking metrics such as MRR, Recall@k, nDCG@k, or MAP.
- Some cells are exploratory ablations and earlier versions of MAG. The final corrected implementation is the last cell under **Upgraded MAG**.
- The notebook should be executed sequentially within each dataset block because later cells depend on JSON files generated by earlier preparation cells.
- For formal reproducibility, package versions and random seeds should be fixed. Some libraries, especially Gensim LDA and neural embedding models, can show small variations across environments.

## Summary

The notebook demonstrates that MAG is a sparse retrieval framework that improves BM25 by changing the indexed representation from raw passage text to a fused metadata representation. The framework extracts metadata from entities, nouns, latent topic terms, semantic keyphrases, and statistical keywords, then maps these metadata scores into a BM25-compatible weighted keyword corpus.

Across the reported experiments, the strongest MAG variants achieve competitive or improved top-1 retrieval accuracy compared with raw BM25 while preserving fast sparse retrieval. The final corrected MAG implementation is especially effective on the SQuAD passage retrieval setup, where it improves accuracy from 0.7141 for BM25 to 0.7438 with sub-millisecond average retrieval time in the recorded run.
