from dataclasses import asdict
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from forgebuilder.pipeline import CriticResult, VerificationResult
from forgebuilder.schema import sha256_bytes
from forgebuilder.serialization import serialize_posttraining
from forgebuilder.trajectory import normalize_trajectory


def fixture():
    tokenizer=Tokenizer(models.WordLevel({'[UNK]':0,'<bos>':1,'<eos>':2,'good':3,'bad':4,'答':5},unk_token='[UNK]'))
    tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
    record=dict(sample_id='fixture',accepted=True,license='MIT',eval_quarantine_status='clear_lexical_screen',
        candidate='good 答',content_sha256=sha256_bytes('good 答'.encode()),
        task={'instruction':'Inspect bad evidence.','kind':'fixture','evidence':{'excerpt':'bad'}},
        critic=asdict(CriticResult('PASS',.9)),
        objective_verification=asdict(VerificationResult('PASS','fixture','v1','a'*64,'b'*64,passed=1)))
    return record,tokenizer


def test_known_bad_and_observations_are_context_only_with_aligned_eos():
    record,tokenizer=fixture()
    record['trajectory']=normalize_trajectory([
        {'type':'WRONG_HYPOTHESIS','content':'bad 答','loss_mask':1,'target_weight':1},
        {'type':'TOOL_RESULT','content':'bad','loss_mask':1},
        {'type':'FINAL','content':record['candidate']}])
    result=serialize_posttraining(record,tokenizer)
    assert result['input_ids'][0]==1 and result['target_weights'][0]==0
    assert result['input_ids'][-1]==2 and result['target_weights'][-1]==1
    assert result['text'].endswith(record['candidate'])
    assert all(w==0 for token,w in zip(result['input_ids'],result['target_weights']) if token==4)
    assert result['target_weights'].count(1.)==3  # good, final Unicode answer, EOS
    assert result['frozen_training_recipe'] is False
    assert serialize_posttraining(record,tokenizer)['serialized_sha256']==result['serialized_sha256']


def test_hidden_oracles_are_not_serialized_as_user_context():
    record,tokenizer=fixture()
    record['task'].update(kind='new_project_function',evidence={'specification':'Implement a new helper.',
                                                             'cases':[{'expected':'PRIVATE_ORACLE'}]})
    result=serialize_posttraining(record,tokenizer)
    assert 'PRIVATE_ORACLE' not in result['text']


def test_terminal_observation_only_appears_after_the_tool_call():
    record,tokenizer=fixture()
    record['task'].update(kind='rg_definition_locations',evidence={'excerpt':'good',
        'terminal_execution':{'stdout':'OBSERVED_ONLY_LATER'}})
    record['trajectory']=normalize_trajectory([
        {'type':'TOOL_CALL','content':'rg query source.py'},
        {'type':'TOOL_RESULT','content':'OBSERVED_ONLY_LATER'},
        {'type':'FINAL','content':record['candidate']}])
    result=serialize_posttraining(record,tokenizer)
    assert 'OBSERVED_ONLY_LATER' not in result['text'].split('[TOOL_CALL]')[0]
    assert result['text'].count('OBSERVED_ONLY_LATER')==1


@pytest.mark.parametrize('change',['unknown_license','pending','changed_candidate','bad_final','critic_issue'])
def test_unaccepted_or_changed_records_cannot_be_serialized(change):
    record,tokenizer=fixture()
    if change=='unknown_license': record['license']=None
    elif change=='pending': record['accepted']=None
    elif change=='changed_candidate': record['candidate']='changed'
    elif change=='bad_final': record['trajectory']=normalize_trajectory([{'type':'FINAL','content':'bad'}])
    elif change=='critic_issue': record['critic']['hallucination']=True
    with pytest.raises(ValueError): serialize_posttraining(record,tokenizer)


def test_fractional_weights_are_retained():
    record,tokenizer=fixture()
    record['trajectory']=normalize_trajectory([{'type':'FINAL','content':record['candidate'],'target_weight':.25}])
    result=serialize_posttraining(record,tokenizer)
    assert result['target_weights'][-3:]==[.25,.25,1.]


def test_token_crossing_context_target_boundary_cannot_be_a_target():
    record,_=fixture()
    class MergingTokenizer:
        def encode(self,text,**kwargs):
            return SimpleNamespace(ids=[3],offsets=[(0,len(text))])
        def token_to_id(self,token): return {'<bos>':1,'<eos>':2}[token]
    with pytest.raises(ValueError,match='no content target'):
        serialize_posttraining(record,MergingTokenizer())
