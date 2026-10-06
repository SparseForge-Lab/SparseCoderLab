"""Loopback-only CPU playground for the retained Dense50M base checkpoint."""
from __future__ import annotations
import json,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from src.config import load_config
from src.model import LanguageModel
from src.training.research import sha

ROOT=Path(__file__).resolve().parents[1]
PAGE=ROOT/'web/local_dense_chat.html'
LOCK=threading.Lock()
torch.set_num_threads(2);torch.set_num_interop_threads(1)
CFG=load_config(str(ROOT/'configs/research_v1/dense.yaml'))
CHECKPOINT=ROOT/'experiments/research_v1/dense/checkpoints/tokens_50003968.pt'
SUMMARY=json.loads((CHECKPOINT.parent.parent/'summary_50003968.json').read_text(encoding='utf-8'))
assert sha(CHECKPOINT)==SUMMARY['checkpoint_sha256']
MODEL=LanguageModel(CFG).cpu().eval()
STATE=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
assert STATE['metadata']['tokens_seen']==50003968
MODEL.load_state_dict(STATE['model'],strict=True);del STATE
TOKENIZER=Tokenizer.from_file(str(ROOT/CFG['data']['tokenizer']))
assert sha(ROOT/CFG['data']['tokenizer'])==SUMMARY['tokenizer_hash']
INFO={'model':'DenseCompute · 50M','tokens':50003968,'parameters':SUMMARY['stored_params'],
      'device':'CPU','threads':2,'context':1024,'checkpoint_sha256':SUMMARY['checkpoint_sha256'],
      'type':'Base language model; no chat/instruction fine-tuning'}

def next_logits(ids):
    # Same frozen dense backbone; project only the last hidden position to avoid
    # computing discarded full-vocabulary logits for every prompt token on CPU.
    x=MODEL.embedding(ids)
    for block in MODEL.blocks:x,_=block(x)
    return F.linear(MODEL.norm(x[:,-1]),MODEL.embedding.weight)[0]

class Handler(BaseHTTPRequestHandler):
    def reply(self,status,body,kind='application/json; charset=utf-8'):
        if isinstance(body,dict):body=json.dumps(body).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
        self.end_headers();self.wfile.write(body)
    def do_GET(self):
        if self.path=='/':self.reply(200,PAGE.read_bytes(),'text/html; charset=utf-8')
        elif self.path=='/info':self.reply(200,INFO)
        else:self.reply(404,{'error':'Not found'})
    def do_POST(self):
        if self.path!='/generate':return self.reply(404,{'error':'Not found'})
        if self.headers.get('Origin') not in (None,'http://127.0.0.1:8765','http://localhost:8765'):
            return self.reply(403,{'error':'Use the local playground page'})
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=64000:raise ValueError('Prompt request too large')
            request=json.loads(self.rfile.read(length));prompt=request['prompt']
            if not isinstance(prompt,str) or not prompt.strip():raise ValueError('Enter a prompt')
            maximum=max(1,min(256,int(request.get('max_tokens',96))))
            temperature=max(0.,min(1.5,float(request.get('temperature',.7))))
            seed=int(request.get('seed',42))
        except Exception as error:return self.reply(400,{'error':str(error)})
        if not LOCK.acquire(blocking=False):return self.reply(409,{'error':'Another generation is running; try again shortly'})
        try:
            ids=[TOKENIZER.token_to_id('<bos>')]+TOKENIZER.encode(prompt).ids
            truncated=len(ids)>1024;ids=ids[-1024:]
            x=torch.tensor([ids],dtype=torch.long,device='cpu');generated=[]
            generator=torch.Generator(device='cpu').manual_seed(seed)
            stop={TOKENIZER.token_to_id('<eos>'),TOKENIZER.token_to_id('<doc>')}
            self.send_response(200);self.send_header('Content-Type','application/x-ndjson; charset=utf-8')
            self.send_header('Cache-Control','no-store');self.send_header('Connection','close');self.end_headers()
            start=time.perf_counter();reason='token limit'
            with torch.inference_mode():
                for _ in range(maximum):
                    logits=next_logits(x)
                    if temperature==0:value=int(logits.argmax())
                    else:
                        values,indices=torch.topk(logits/temperature,40)
                        value=int(indices[torch.multinomial(torch.softmax(values,0),1,generator=generator)])
                    if value in stop:reason='end token';break
                    generated.append(value)
                    text=TOKENIZER.decode(generated,skip_special_tokens=True)
                    self.wfile.write((json.dumps({'text':text})+'\n').encode());self.wfile.flush()
                    x=torch.cat((x,torch.tensor([[value]],dtype=torch.long)),1)[:,-1024:]
            self.wfile.write((json.dumps({'done':True,'text':TOKENIZER.decode(generated,skip_special_tokens=True),
                'tokens':len(generated),'seconds':time.perf_counter()-start,'reason':reason,'prompt_truncated':truncated})+'\n').encode());self.wfile.flush()
        except (BrokenPipeError,ConnectionResetError):pass
        finally:LOCK.release()
    def log_message(self,format,*args):
        print(f'{self.command} {self.path}: {format % args}',flush=True)

if __name__=='__main__':
    server=ThreadingHTTPServer(('127.0.0.1',8765),Handler)
    print(json.dumps({'ready':True,'url':'http://127.0.0.1:8765',**INFO}),flush=True)
    server.serve_forever()
