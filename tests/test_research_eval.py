from src.eval.micro_code import check
from tools.research_collect import paired

def test_micro_numeric_checks_and_no_code_execution():
    assert check('def example(a, b):\n    return a + b\n',[(2,3,5)])==(True,True)
    assert check('def example(a, b):\n    return a - b\n',[(2,3,5)])==(True,False)
    assert check('def example(a, b):\n    return __import__("os").system("anything")\n',[(2,3,5)])==(False,False)
def test_paired_document_delta_identity_and_weighting():
    a={'documents':[dict(sha256='one',stratum='code/Python',predictions=100,nll=2.),dict(sha256='two',stratum='code/Python',predictions=200,nll=3.)]}
    b={'documents':[dict(sha256='one',stratum='code/Python',predictions=100,nll=1.),dict(sha256='two',stratum='code/Python',predictions=200,nll=2.)]}
    r=paired(a,b)['code'];assert r['weighted_delta']==-1 and r['approx_95_interval']==[-1,-1]
