"""Tokenize accepted posttraining records with explicit per-token target weights.

This is an offline serialization adapter, not a training loop or frozen recipe.
"""
import math

from .pipeline import CriticResult, VerificationResult, accept_posttraining_candidate
from .schema import canonical_json, sha256_bytes
from .trajectory import normalize_trajectory

CONTEXT_EVENTS={'EVIDENCE','OBSERVATION','TOOL_RESULT','FAILURE_OUTPUT','WRONG_HYPOTHESIS'}


def serialize_posttraining(record,tokenizer):
    if not record.get('accepted') or not record.get('license'):
        raise ValueError('accepted record with known license metadata required')
    if record.get('eval_quarantine_status')!='clear_lexical_screen':
        raise ValueError('record has not cleared the registered quarantine screen')
    proof=VerificationResult(**record['objective_verification'])
    critic=CriticResult(**record['critic'])
    if not accept_posttraining_candidate(verification=proof,critic=critic):
        raise ValueError('objective and independent critic acceptance required')
    if any(getattr(critic,k) for k in ('functional_issue','hallucination','scope_violation',
                                      'stop_issue','reasoning_issue','test_gaming','needs_deeper_review')):
        raise ValueError('unresolved critic issue')
    candidate=record['candidate']
    if not candidate or sha256_bytes(candidate.encode())!=record['content_sha256']:
        raise ValueError('candidate content changed or is empty')
    task=record['task']
    evidence=task['evidence']
    if task['kind']=='new_project_function':
        # Hidden qualification oracles are verifier/critic evidence, not part
        # of the generator's function specification or serialized user input.
        evidence={'specification':evidence['specification']}
    parts=[]; spans=[]; position=0

    def add(text,weight,kind):
        nonlocal position
        if not math.isfinite(weight) or weight<0:
            raise ValueError('target weight must be finite and non-negative')
        parts.append(text)
        spans.append({'start':position,'end':position+len(text),'weight':weight,'kind':kind})
        position+=len(text)

    add('[TASK]\n'+task['instruction']+'\n[EVIDENCE]\n'+canonical_json(evidence).decode()+'\n',0.,'task_context')
    if 'trajectory' in record:
        trajectory=normalize_trajectory(record['trajectory']['events'])
        if trajectory['events'][-1]['content']!=candidate:
            raise ValueError('trajectory final result differs from accepted candidate')
        events=trajectory['events']
    else:
        events=[{'type':'FINAL','content':candidate,'loss_mask':1.,'target_weight':1.,'known_bad':False}]
    for index,event in enumerate(events):
        add('['+event['type']+']\n',0.,'event_header')
        weight=event['loss_mask']*event['target_weight']
        if event['type'] in CONTEXT_EVENTS or event['known_bad']: weight=0.
        add(event['content'],weight,event['type'])
        if index+1<len(events):
            add('\n',0.,'event_separator')
    text=''.join(parts)
    encoded=tokenizer.encode(text,add_special_tokens=False)
    weights=[]; span_index=0
    for start,end in encoded.offsets:
        while span_index+1<len(spans) and spans[span_index]['end']<=start:
            span_index+=1
        span=spans[span_index]
        # One-shot tokenization can merge across mask boundaries. Such a token
        # stays context-only so no part of a known-bad span becomes a target.
        weights.append(span['weight'] if end>start and span['start']<=start<end<=span['end'] else 0.)
    bos,eos=(tokenizer.token_to_id(token) for token in ('<bos>','<eos>'))
    if bos is None or eos is None:
        raise ValueError('frozen tokenizer is missing existing BOS/EOS tokens')
    ids=[bos]+encoded.ids+[eos]
    weights=[0.]+weights+[1.]
    if not any(weight>0 for weight in weights[1:-1]):
        raise ValueError('record contains no content target tokens')
    return dict(schema_version=1,sample_id=record['sample_id'],
                serialization_recipe='posttraining_target_spans_v1',text=text,spans=spans,
                input_ids=ids,target_weights=weights,
                token_count=len(ids),target_token_count=sum(w>0 for w in weights),
                serialized_sha256=sha256_bytes(canonical_json({'input_ids':ids,'target_weights':weights})),
                loss_alignment='Predict input_ids[1:] from input_ids[:-1] using target_weights[1:].',
                frozen_training_recipe=False)
