from tools.prompt3_train import atomic_copy


def test_prompt3_checkpoint_copy_is_atomic(tmp_path):
    source = tmp_path / "source.bin"
    target = tmp_path / "nested" / "target.bin"
    source.write_bytes(b"checkpoint metadata")

    atomic_copy(source, target)

    assert target.read_bytes() == source.read_bytes()
    assert not target.with_suffix(".bin.tmp").exists()
