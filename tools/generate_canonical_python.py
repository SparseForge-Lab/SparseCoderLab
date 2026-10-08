"""Fixed coding generation from SHA-verified immutable model-only checkpoints."""
from __future__ import annotations
import argparse
import gc
import json
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from src.config import load_config
from src.model import LanguageModel
from src.utils.hashing import sha256_file


def gpu_allowed():
    control=Path('.local/process_controls/gpu_pause.json')
    if control.exists() and json.loads(control.read_text())['gpu_work_paused']:
        raise RuntimeError('User GPU pause is active; preserving partial output')


def last_logits(model, current, lengths):
    """Same causal transformer; project only the real final hidden position."""
    hidden=model.embedding(current)
    for index,block in enumerate(model.blocks):
        if model.memory is not None and index==model.cfg['memory']['layer']:
            hidden=model.memory(hidden,current)
        hidden,_=block(hidden)
    selected=hidden[torch.arange(current.shape[0],device=current.device),lengths]
    return F.linear(model.norm(selected),model.embedding.weight)


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts, temperature, maximum=128):
    sequences=[[tokenizer.token_to_id('<bos>')]+tokenizer.encode(prompt).ids for prompt in prompts]
    if any(len(ids)+maximum>1024 for ids in sequences): raise ValueError('Fixed prompt exceeds context budget')
    generators=[torch.Generator(device='cuda').manual_seed(42) for _ in prompts]
    stop={tokenizer.token_to_id('<eos>'),tokenizer.token_to_id('<doc>')}
    results=[[] for _ in prompts]; finished=[False for _ in prompts]
    for _ in range(maximum):
        # Right padding follows the real last-token index. Causal attention and
        # causal Ngram lookup cannot use those future placeholders. MoE drops no
        # tokens and its load auxiliary loss does not modify inference outputs.
        current=torch.zeros((len(prompts),max(map(len,sequences))),dtype=torch.long,device='cuda')
        for row,ids in enumerate(sequences):current[row,:len(ids)]=torch.tensor(ids,device='cuda')
        lengths=torch.tensor([len(ids)-1 for ids in sequences],device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=last_logits(model,current,lengths).float()
        for row in range(len(prompts)):
            if finished[row]:continue
            if temperature==0:value=int(logits[row].argmax())
            else:
                values,indices=torch.topk(logits[row]/temperature,40)
                probabilities=torch.softmax(values,0);cumulative=probabilities.cumsum(0)
                probabilities=probabilities.masked_fill((cumulative-probabilities)>=.95,0)
                probabilities/=probabilities.sum()
                value=int(indices[torch.multinomial(probabilities,1,generator=generators[row])])
            if value in stop:finished[row]=True
            else:results[row].append(value);sequences[row].append(value)
        if all(finished):break
    return [(tokenizer.decode(result,skip_special_tokens=True),len(result)) for result in results]


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.batch_size<=8:raise ValueError('Evaluation batch must be between1 and8')
    if args.output.exists(): raise FileExistsError('Use a new generation version')
    gpu_allowed(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    manifest=json.loads(Path('results/prompt3/transition_checkpoint_manifest.json').read_text(encoding='utf8'))
    benchmark=json.loads(Path('configs/eval/simple_python_v1.json').read_text(encoding='utf8'))
    tokenizer=Tokenizer.from_file('data/research_v1/tokenizer.json')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter(); count=0
    with args.output.open('x',encoding='utf8',newline='\n') as handle:
        for tag,entry in manifest['canonical_transition_checkpoints'].items():
            checkpoint=Path(entry['model_weights']['path']); expected=entry['model_weights']['sha256']
            if sha256_file(checkpoint)!=expected: raise ValueError('Canonical model hash mismatch')
            cfg=load_config(entry['config']['path'])
            if sha256_file(Path(cfg['data']['tokenizer']))!=entry['tokenizer_sha256']: raise ValueError('Tokenizer changed')
            model=LanguageModel(cfg).cuda().eval()
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
            for temperature in (0.,.2,.5,.8,1.,1.2,1.5):
                gpu_allowed();begin=time.perf_counter()
                generated=[]
                for offset in range(0,len(benchmark['tasks']),args.batch_size):
                    gpu_allowed()
                    generated.extend(generate_batch(model,tokenizer,
                        [task['prompt'] for task in benchmark['tasks'][offset:offset+args.batch_size]],temperature))
                elapsed=time.perf_counter()-begin
                for task,(text,tokens) in zip(benchmark['tasks'],generated):
                    row=dict(variant=tag,temperature=temperature,prompt_number=task['prompt_number'],
                        input=task['prompt'],raw_output=text,checkpoint=checkpoint.as_posix(),checkpoint_sha256=expected,
                        training_tokens=100_007_936,generation_seed=42,max_new_tokens=128,top_p=.95,top_k=40,
                        repetition_penalty=1.,stop_tokens=['<eos>','<doc>'],device='cuda',bf16=True,tf32=True,
                        generation_group_seconds=elapsed,generation_batch_size=args.batch_size,generated_tokens=tokens,
                        sampler_sha256=sha256_file(Path(__file__)))
                    handle.write(json.dumps(row,ensure_ascii=False)+'\n'); handle.flush(); count+=1
                print(json.dumps(dict(variant=tag,temperature=temperature,completions=count,
                    allocated_mib=torch.cuda.memory_allocated()/1024**2,reserved_mib=torch.cuda.memory_reserved()/1024**2,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)),flush=True)
                torch.cuda.empty_cache()
            if sha256_file(checkpoint)!=expected: raise ValueError('Canonical model changed during generation')
            del model;gc.collect();torch.cuda.empty_cache()
    print(json.dumps(dict(completed=count,seconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__': main()
