import argparse,json
from pathlib import Path
from src.moe.trace import read_trace
from src.moe.atlas import expert_atlas,pack_experts
from src.runtime.scaling import scale_trace
from src.runtime.cache import IOConfig,simulate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--source-experts',type=int,default=12);p.add_argument('--capacity-layers',type=int,default=16)
    p.add_argument('--experts',type=int,default=256);p.add_argument('--expert-bytes',type=int,default=5898240);p.add_argument('--page-experts',type=int,default=2)
    p.add_argument('--vram-gib',type=float,default=4);p.add_argument('--ram-gib',type=float,default=32);p.add_argument('--ssd-mib-s',type=float,required=True)
    p.add_argument('--ssd-latency-ms',type=float,default=.2);p.add_argument('--rotate-tokens',type=int,default=1024);a=p.parse_args()
    raw=list(read_trace(a.trace));records=scale_trace(raw,a.source_experts,a.capacity_layers,a.experts,a.rotate_tokens);cut=max(len(records)//2,1)
    atlas=expert_atlas(records[:cut],a.experts)
    for layer in range(a.capacity_layers):
        for expert in range(a.experts):atlas['load'].setdefault(layer*a.experts+expert,0)
    mapping=pack_experts(atlas,a.page_experts)
    io=IOConfig(a.expert_bytes,int(a.vram_gib*1024**3),int(a.ram_gib*1024**3),a.ssd_mib_s*1024**2,a.ssd_latency_ms/1000,12000*1024**2,.00002)
    result={'inputs':vars(a)|{'trace':str(a.trace)},'simulation':simulate(records[cut:],mapping,a.experts,io),
            'scope':'Synthetic replicated layers and rotating expert identities; a sensitivity scenario, NOT observed large-model routing or speed.'}
    Path('results/scaled_simulation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
