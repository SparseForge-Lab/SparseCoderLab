"""Test grouped real-stream BF16 interrupted/continuous persistence before quality."""
from __future__ import annotations
import copy,hashlib,json,os,tempfile
from pathlib import Path
import torch
from src.config import load_config
from src.model import LanguageModel
from src.training.engine import require_cuda,make_optimizer,amp,seed_all
from src.training.checkpoint import save_training,resume_training
from src.training.data import PackedStream
from src.training.research import RESULTS,training_hash,frozen_identity,checkpoint_identity
from src.eval.research import freeze_evaluation

def main():
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True);require_cuda('cuda');seed_all(42)
    cfg=load_config('configs/research_v1/memory.yaml');freeze_evaluation(cfg)
    model=LanguageModel(cfg).cuda();opt,scheduler=make_optimizer(model,cfg);stream=PackedStream(Path(cfg['data']['shards']),'train',1024,42)
    def step(m,o,s,loader):
        m.train();o.zero_grad();inputs=[]
        for _ in range(4):
            x,y=loader.next(2,'cuda');inputs.append(hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest())
            with amp(cfg):loss=m(x,y)['loss']/4
            loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);o.step();s.step();return inputs
    step(model,opt,scheduler,stream)
    meta=dict(step=1,tokens_seen=8192,cursor=stream.cursor,**frozen_identity(cfg))
    with tempfile.TemporaryDirectory(dir=Path('work')) as directory:
        path=Path(directory)/'resume.pt';save_training(path,model,opt,scheduler,meta)
        expected_inputs=step(model,opt,scheduler,stream);expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};expected_opt=copy.deepcopy(opt.state_dict());expected_schedule=copy.deepcopy(scheduler.state_dict())
        restored=LanguageModel(cfg).cuda();other,other_scheduler=make_optimizer(restored,cfg)
        loaded=resume_training(path,restored,other,other_scheduler,'cpu');checkpoint_identity(loaded,cfg)
        loader=PackedStream(Path(cfg['data']['shards']),'train',1024,42,loaded['cursor']);actual_inputs=step(restored,other,other_scheduler,loader)
        max_difference=max(float((v-restored.state_dict()[k].cpu()).abs().max()) for k,v in expected.items())
        optimizer_equal=all(torch.equal(value.cpu(),other.state_dict()['state'][key][field].cpu()) if torch.is_tensor(value) else value==other.state_dict()['state'][key][field]
                            for key,state in expected_opt['state'].items() for field,value in state.items())
        result=dict(passed=actual_inputs==expected_inputs and max_difference==0 and optimizer_equal and expected_schedule==other_scheduler.state_dict() and loader.cursor==stream.cursor,
                    identical_inputs=actual_inputs==expected_inputs,max_weight_difference=max_difference,optimizer_equal=optimizer_equal,scheduler_equal=expected_schedule==other_scheduler.state_dict(),
                    cursor_before=loaded['cursor'],cursor_after=loader.cursor,token_count_before=loaded['tokens_seen'],source_hash=training_hash(),
                    scope='Actual width320/nine-layer grouped+Ngram model, frozen research stream, four BF16 microbatches/step. Deterministic process-local algorithms verify persistence; no quality checkpoint is retained.')
        (RESULTS/'resume_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
        if not result['passed']:raise RuntimeError('Real-data resume verification failed')
if __name__=='__main__':main()
