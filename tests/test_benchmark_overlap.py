from tools.screen_benchmark_overlap import TokenMatcher, patterns


def test_token_matcher_shared_prefix_failure_links_suffixes_and_no_substrings():
    matcher=TokenMatcher([(['a','b','c'],('one','ref')),(['b','c'],('two','ref')),
        (['a','b','d'],('three','ref')),(['c','d'],('four','ref'))])
    assert matcher.matches(['x','a','b','c','d','a','b','d']) == [('four','ref'),('one','ref'),('three','ref'),('two','ref')]
    assert matcher.matches(['aa','b','cc']) == []


def test_overlap_profiles_skip_short_sequences_and_use_humaneval_docstring():
    registry={'tasks':[dict(id='short',prompt='import os\ndef f():\n    """Short description."""',reference='return 0'),
        dict(id='long',prompt=' '.join('word'+str(i) for i in range(12)),reference=' '.join('token'+str(i) for i in range(32)))]}
    entries,profiles=patterns(registry)
    assert len(entries)==2
    assert profiles[0]['prompt_tokens']==3 and not profiles[0]['reference_screened']
    assert profiles[1]['reference_screened'] and profiles[1]['prompt_screened']
