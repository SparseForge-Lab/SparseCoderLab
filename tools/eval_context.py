"""Neural answer NLL at each compaction boundary; oracle retrieval is explicit."""
import argparse, json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from torch.nn import functional as F
from src.config import load_config
from src.context.archive import Archive, compact_fields
from src.context.tasks import make_task
from src.model import LanguageModel
from src.training.engine import require_cuda, amp

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config',default='configs/dense_compute.yaml'); p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--tasks',type=int,default=2); p.add_argument('--context',type=int,choices=[1024,2048,4096,8192,16384],default=8192)
    a = p.parse_args(); cfg=load_config(a.config); require_cuda('cuda'); model=LanguageModel(cfg).cuda().eval()
    model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model']); tokenizer=Tokenizer.from_file(cfg['data']['tokenizer']); archive=Archive(Path('data/archive')); rows=[]
    for index in range(a.tasks):
        task=make_task(index,cfg,archive); answer=json.dumps(task['facts'],sort_keys=True); answer_ids=tokenizer.encode(answer).ids
        for ratio in cfg['context']['ratios']:
            fields=dict(task['facts'])
            for boundary in range(cfg['context']['boundaries']+1):
                if boundary: fields=compact_fields(fields,task['pointers'],ratio)['fields']
                for retrieval in (False,True):
                    prompt=task['prompt'] if boundary==0 or ratio==1 else '<compact>'+json.dumps({'fields':fields,'pointers':task['pointers']})
                    if retrieval: prompt+='\n<retrieved>'+''.join(archive.get(pointer) for pointer in task['pointers'])
                    prompt+='\n'+task['question']+'\nANSWER:\n'; prefix=tokenizer.encode(prompt).ids
                    original_length=len(prefix); room=a.context-len(answer_ids)
                    if room<1: raise ValueError('Answer does not fit context')
                    # Keep initial facts plus final question/retrieval; document middle truncation.
                    if len(prefix)>room: prefix=prefix[:room//2]+prefix[-(room-room//2):]
                    sequence=torch.tensor([prefix+answer_ids],device='cuda')
                    with torch.no_grad(),amp(cfg): logits=model(sequence[:,:-1])['logits'][:,len(prefix)-1:]
                    targets=torch.tensor([answer_ids],device='cuda'); loss=F.cross_entropy(logits.float().flatten(0,1),targets.flatten())
                    accuracy=(logits.argmax(-1)==targets).float().mean()
                    rows.append({'task':index,'ratio':ratio,'boundary':boundary,'oracle_retrieval':retrieval,'answer_loss':float(loss),'teacher_forced_answer_token_accuracy':float(accuracy),
                                 'prompt_tokens_before_truncation':original_length,'prompt_tokens_used':len(prefix),'context':a.context,'truncation_policy':'head/tail halves; omitted middle tokens explicitly recorded',
                                 'note':'Answer likelihood/teacher-forced token accuracy, not generated task pass rate. Retrieval selection is oracle.'})
    Path('results/context_model_eval.json').write_text(json.dumps(rows,indent=2)); print(f'{len(rows)} measured model/task states written.')
