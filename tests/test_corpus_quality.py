from tools.finalize_repository_corpus import quality_reason


def test_quality_policy_holds_empty_and_extreme_repetition_but_retains_normal_code():
    assert quality_reason(' \n')=='low_information_under64_nonwhitespace_chars'
    assert quality_reason('the same sufficiently long line\n'*32)=='repetitive_over80percent_identical_nonempty_lines'
    assert quality_reason('!!! '*1024)=='low_information_under2percent_alphanumeric'
    assert quality_reason('def bounded_total(values):\n    result = 0\n    for value in values:\n        result += value\n    return result\n') is None
