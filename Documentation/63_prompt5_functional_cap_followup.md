# Prompt-5 functional cap follow-up

I rescored the frozen Prompt-5 completions after making generation truncation an independent flag. The scorer now extracts, parses, and executes the available candidate even when it reaches its token cap. The official historical 128-token report remains unchanged; these are separate diagnostic reports using the recovered historical benchmark fixture.

The complete 128-token set contains 400 completions: 100 each from `dense75_ref`, `sparse75`, `sparse75_ngram10m`, and `sparse75_ngram25m`. The all-model 256-token set also contains 400 completions. A separate 512-token generation was stopped at the user's request after 216 rows, so its report is explicitly partial and does not include `sparse75_ngram25m`.

| Generation cap | Variant | Functional passes | Parseable | Truncated |
|---:|---|---:|---:|---:|
| 128 | `dense75_ref` | 0/100 | 27 | 86 |
| 128 | `sparse75` | 0/100 | 12 | 96 |
| 128 | `sparse75_ngram10m` | 2/100 | 31 | 75 |
| 128 | `sparse75_ngram25m` | 1/100 | 18 | 98 |
| 256 | `dense75_ref` | 0/100 | 24 | 86 |
| 256 | `sparse75` | 5/100 | 21 | 96 |
| 256 | `sparse75_ngram10m` | 2/100 | 30 | 75 |
| 256 | `sparse75_ngram25m` | 1/100 | 13 | 98 |
| 512, partial | `dense75_ref` | 3/100 | 29 | 86 |
| 512, partial | `sparse75` | 4/100 | 24 | 96 |
| 512, partial | `sparse75_ngram10m` | 1/16 | 8 | 8 |

The five functional passes in the sparse 256-token group all reached the cap and also carried the repetition flag; the WASI evaluator nevertheless ran them and checked the requested function against the frozen cases. The 128-token complete run produced three functional passes overall. Parseability is not correctness: many parseable candidates returned wrong values or raised errors, and most capped outputs remained syntax-invalid or repetitive.

The 512-token counts are exploratory because the run ended before all four variants completed. They should not be treated as a matched four-model comparison. The 1,024-token generation request does not fit the unchanged 1,024-token model context for the longest prompt; the uniform safe generation budget would be 979 tokens. No 500M continuation was started based on this follow-up.

The complete score reports retain each raw completion, its checkpoint and benchmark hashes, per-case execution results, truncation flags, and repetition metrics. The partial 512-token score report records the extracted candidate and classification; its paired generation JSONL retains the raw text:

- `results/unified_python_250M_functional_128tok_truncation_scored_r3.txt`
- `results/unified_python_250M_functional_256tok_truncation_scored_r2.txt`
- `results/unified_python_250M_functional_512tok_partial_truncation_scored_r1.txt`

All scoring used the CPU-only, hash-verified WASI runtime. The updated unified evaluator test file passed 11 tests. These small fixed-case results answer whether any saved output can work; they do not establish broad coding ability or justify a model-quality ranking.
