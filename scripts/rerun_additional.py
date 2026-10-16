"""Run only semantic or timing experiments actually discussed in the paper."""
from pathlib import Path
import argparse, sys, yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from stride.config import RouterConfig
from stride.models.factory import create_adapter
from stride.evaluation.runner import evaluate_jsonl
def main():
    p=argparse.ArgumentParser();p.add_argument('study',choices=['semantic','timing']);p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    spec=yaml.safe_load((ROOT/'configs'/f'{args.study}.yaml').read_text())
    if args.dry_run:print(spec);return
    base=RouterConfig.from_yaml(ROOT/'configs/router.yaml')
    from stride.reproducibility import seed_everything
    seed_everything(base.seed)
    adapter=create_adapter('llava15-7b',base,device='cuda',dtype='bfloat16',load_in_4bit=True)
    for job in spec['jobs']:
        adapter.router_config=RouterConfig.from_dict({**base.to_dict(),'budget':job['budget'],'stride_use_semantics':args.study!='semantic','seed':0})
        kwargs={'max_new_tokens':12}
        if args.study=='timing':kwargs['measure_decoder_timing']=True
        evaluate_jsonl(adapter,ROOT/'data_evaluation'/job['task']/'samples.jsonl',ROOT/'runs'/args.study/job['task']/f"{job['method']}_b{job['budget']}",method=job['method'],budget=job['budget'],max_samples=job['samples'],generation_kwargs=kwargs,seed=0)
if __name__=='__main__':main()
