# Frozen code-friendly tokenizer

The tokenizer is byte-level BPE with exactly 32,768 vocabulary entries. Its initial alphabet covers all 256 UTF-8 byte values, so unseen Unicode can be reconstructed. `byte_fallback=True` is also set; byte-level alphabet coverage provides the roundtrip guarantee. No whitespace normalizer is used. The ByteLevel pretokenizer disables inserted prefix spaces and the ByteLevel decoder restores original bytes.

Reserved tokens: `<bos>`, `<eos>`, `<doc>`, `<repo>`, `<file>`, `<tool_call>`, `<tool_result>`, `<compact>`, `<mem>`, `<retrieved>`. Archive pointers serialize as `<mem:000123>`; variable pointer IDs use regular BPE pieces rather than allocating a token for every ID.

`python -m tools.prepare_data` trains only on the training split, saves `data/tokenizer.json` once and freezes it for all comparisons. If the corpus cannot fill the exact requested vocabulary, preprocessing fails instead of shrinking the model silently. Reusing existing tokenizer files never retrains them.

`results/tokenizer_quality.json` reports tokens/character and tokens/line for Python, C, C++, Rust, Java, Go, JavaScript, TypeScript, HTML/CSS, SQL, Bash, JSON/YAML and English. Fixtures also check tabs, CRLF, Unicode, NUL and reserved markers. These samples test encoding, not broad language representativeness. The manifest and checkpoint record the tokenizer SHA256.
