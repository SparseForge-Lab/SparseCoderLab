import argparse,json
from pathlib import Path
from src.config import load_config
from src.memory.ngram import NgramMemory
from src.training.data import PackedStream

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/sparse_memory.yaml');p.add_argument('--tokens',type=int,default=4096);a=p.parse_args();cfg=load_config(a.config)
    stream=PackedStream(Path(cfg['data']['shards']),'train',min(a.tokens,cfg['training']['context']),cfg['data']['seed'])
    tokens,_=stream.next(max(a.tokens//stream.context,1),'cpu');memory=NgramMemory(cfg['memory'],cfg['model']['d_model']);report=memory.statistics(tokens)
    Path('results/ngram_statistics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
