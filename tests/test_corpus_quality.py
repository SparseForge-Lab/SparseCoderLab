import pytest
from tools.finalize_repository_corpus import quality_reason,apply_reviewed_metadata


def test_quality_policy_holds_empty_and_extreme_repetition_but_retains_normal_code():
    assert quality_reason(' \n')=='low_information_under64_nonwhitespace_chars'
    assert quality_reason('the same sufficiently long line\n'*32)=='repetitive_over80percent_identical_nonempty_lines'
    assert quality_reason('!!! '*1024)=='low_information_under2percent_alphanumeric'
    assert quality_reason('def bounded_total(values):\n    result = 0\n    for value in values:\n        result += value\n    return result\n') is None


def test_final_metadata_correction_preserves_body_and_rejects_unreviewed_pins():
    row=dict(repository='fmtlib/fmt',revision='a'*40,source_license_sha256='b'*64,file_path='include/fmt/core.h',language='C',text='original bytes')
    sources={'fmtlib/fmt':dict(revision='a'*40,root_license_sha256='b'*64,header_language='C++')}
    apply_reviewed_metadata(row,sources)
    assert row['language']=='C++' and row['text']=='original bytes'
    row['revision']='c'*40
    with pytest.raises(ValueError,match='pin/license'):apply_reviewed_metadata(row,sources)
