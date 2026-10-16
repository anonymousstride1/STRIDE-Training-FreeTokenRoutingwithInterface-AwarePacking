"""Download public benchmarks, then validate and recover the exact frozen subset."""
from pathlib import Path
import json, subprocess, sys, shutil, hashlib
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
def rows(p):return [json.loads(s) for s in p.open(encoding='utf-8') if s.strip()]
def signature(r):return str(r['id']),r['question'],tuple(r['answers'])
def main():
    for task in ['gqa','mmbench','mme','pope','scienceqa','textvqa']:
        full=ROOT/'work/downloaded'/task
        if not (full/'samples.jsonl').exists():
            subprocess.run([sys.executable,str(ROOT/'scripts/prepare_benchmarks.py'),task,'--output-root',str(ROOT/'work/downloaded'),'--seed','0'],check=True)
        fresh=rows(full/'samples.jsonl'); frozen=rows(ROOT/'data/frozen'/task/'samples.jsonl')
        assert [signature(r) for r in fresh[:2000]]==[signature(r) for r in frozen],f'{task}: upstream order/content changed; do not silently substitute examples'
        output=ROOT/'data_evaluation'/task;(output/'images').mkdir(parents=True,exist_ok=True)
        for reference,new in zip(frozen,fresh):
            src=full/new['image'].replace('\\','/');dest=output/reference['image']
            with Image.open(src) as im:
                digest=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
            assert digest==reference['metadata_pixel_sha256'],f'{task}: upstream pixel content changed'
            shutil.copy2(src,dest)
        payload=[{k:v for k,v in r.items() if k!='metadata_pixel_sha256'} for r in frozen]
        (output/'samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in payload),encoding='utf-8')
        print(task, len(payload),'frozen examples validated')
if __name__=='__main__':main()
