import json

from src.training.engine import read_jsonl_utf8


def test_jsonl_reader_uses_utf8_for_non_ascii_metadata(tmp_path):
    source = tmp_path / "metadata.jsonl"
    source.write_text(json.dumps({"text": "漢字 🚀"}, ensure_ascii=False) + "\n", encoding="utf-8")

    assert read_jsonl_utf8(source) == [{"text": "漢字 🚀"}]
