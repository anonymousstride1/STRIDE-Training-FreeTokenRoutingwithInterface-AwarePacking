"""Regenerate reported artifacts from anonymized, frozen prediction files."""
from pathlib import Path
import argparse, csv, json, sys
from statistics import mean
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from stride.evaluation.benchmark_metrics import summarize_benchmark, mme_complete_pairs
TASKS=['gqa','mmbench','mme','pope','scienceqa','textvqa']
PRIMARY=dict.fromkeys(TASKS,'accuracy')|{'mme':'perception_score','pope':'f1'}
def read(p):
    return [json.loads(s) for s in p.open(encoding='utf-8') if s.strip()]
def csvrows(p):
    return list(csv.DictReader(p.open(encoding='utf-8')))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('artifact',choices=['verify','table1','table2','table3','figure1','intervals','all']);args=parser.parse_args()
    records=csvrows(ROOT/'results/final/main.csv'); lookup={}
    for row in records:
        pred=read(ROOT/row['source'])
        actual=summarize_benchmark(pred,row['task'])[row['metric']]
        assert abs(actual-float(row['value']))<1e-9, (row['source'],actual,row['value'])
        assert len(pred)==2000
        row['value']=actual
        lookup[row['method'],int(row['budget']),row['task']]=row
    ablations=csvrows(ROOT/'results/final/ablations.csv')
    for row in ablations:
        predictions=read(ROOT/row['source'])
        actual=summarize_benchmark(predictions,row['task'])[PRIMARY[row['task']]]
        assert abs(actual-float(row['value']))<1e-9,(row['source'],actual,row['value'])
        full=read(ROOT/lookup['STRIDE',64,row['task']]['source'])
        assert [(r['id'],r['question']) for r in full]==[(r['id'],r['question']) for r in predictions]
        if row['variant'] in ['selection_order','dense_positions']:
            assert all(sorted(a['route_diagnostics']['selected_indices'])==sorted(b['route_diagnostics']['selected_indices']) for a,b in zip(full,predictions)),row['source']
    timing=csvrows(ROOT/'results/final/decoder_timing.csv')
    for row in timing:
        p=ROOT/'results/timing/gqa'/f"{row['method']}_b{row['budget']}/predictions.jsonl"
        predictions=read(p);assert len(predictions)==100
        for field in ['decoder_prefill_seconds','decoder_decode_seconds']:
            assert abs(mean(r[field] for r in predictions)-float(row[field]))<1e-9
    out=ROOT/'artifacts';(out/'tables').mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    def quality(m,k):return mean(100*lookup[m,k,t]['value']/lookup['Dense',576,t]['value'] for t in TASKS)
    def latency(m,k):return mean(float(lookup[m,k,t]['e2e_seconds']) for t in TASKS)
    quality_expected={64:94.80,128:97.65,192:98.51};speed_expected={64:1.80,128:1.57,192:1.44}
    for k in quality_expected:
        assert round(quality('STRIDE',k),2)==quality_expected[k]
        assert round(latency('Dense',576)/latency('STRIDE',k),2)==speed_expected[k]
    for task in ['gqa','mme','textvqa']:
        full=read(ROOT/lookup['STRIDE',64,task]['source']);removed=read(ROOT/'results/semantic'/task/'predictions.jsonl')
        assert [(r['id'],r['question']) for r in full]==[(r['id'],r['question']) for r in removed]
        changed=sum(a['prediction']!=b['prediction'] for a,b in zip(full,removed))
        score=summarize_benchmark(removed,task)[PRIMARY[task]]
        if task=='gqa':assert changed==14 and round(100*score,2)==57.50
        if task=='mme':assert changed==0
        if task=='textvqa':assert round(100*score,2)==44.02
    def table(name,headers,rows,caption,wide=False):
        kind='table*' if wide else 'table'
        value='\n'.join([r'\begin{'+kind+r'}[t]',r'\centering\small',r'\caption{'+caption+'}',r'\begin{tabular}{'+'l'+'r'*(len(headers)-1)+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule',*[' & '.join(map(str,row))+r' \\' for row in rows],r'\bottomrule\end{tabular}',r'\end{'+kind+'}'])+'\n'
        (out/'tables'/name).write_text(value,encoding='utf-8')
    if args.artifact in {'table1','all'}:
        output=[]
        for m,k in [('Dense',576)]+[(m,k) for k in (64,128,192) for m in ['Vision top-k control','DivPrune','VisPruner','STRIDE']]:
            output.append([m,k,*[f"{lookup[m,k,t]['value']*(1 if t=='mme' else 100):.2f}" for t in TASKS],f'{quality(m,k):.2f}'])
        table('main_results.tex',['Method','$K$','GQA','MMBench','MME','POPE','SQA','TextVQA','Rel. (\%)'],output,'Main-backbone matched subsets; MME uses paired perception points and POPE uses F1.',True)
    if args.artifact in {'table2','all'}:
        output=[[('Dense' if r['method']=='dense' else 'STRIDE'),r['budget'],f"{float(r['decoder_prefill_seconds'])*1000:.2f}",f"{float(r['decoder_decode_seconds'])*1000:.2f}"] for r in sorted(timing,key=lambda r:(r['method']!='dense',int(r['budget']))) ]
        table('decoder_timing.tex',['Method','$K$','Prefill (ms)','Decode (ms)'],output,'Synchronized forward times on 100 GQA examples.')
    if args.artifact in {'table3','all'}:
        output=[]
        for t in TASKS:
            values=[lookup['STRIDE',64,t]['value']]+[float(next(r['value'] for r in ablations if r['task']==t and r['variant']==v)) for v in ['salience_only','selection_order','diversity_only','dense_positions']]
            output.append([{'scienceqa':'SQA'}.get(t,t.upper()),*[f"{v*(1 if t=='mme' else 100):.2f}" for v in values]])
        table('ablations.tex',['Task','Full','Sal.','Order','Div.','Positions'],output,'Same-example 64-token ablations; order and positions preserve membership.')
    if args.artifact in {'figure1','all'}:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
        fig,axs=plt.subplots(1,2,figsize=(7,2.3),layout='constrained')
        for m,color,marker in [('STRIDE','#087f8c','o'),('VisPruner','#555555','s'),('DivPrune','#b87827','^'),('Vision top-k control','#73528c','d')]:
            axs[0].plot([64,128,192],[quality(m,k) for k in (64,128,192)],label=m,color=color,marker=marker,lw=1.5)
        axs[0].set(xlabel='Retained visual tokens',ylabel='Relative quality (%)',xticks=[64,128,192]);axs[0].grid(alpha=.2);axs[0].legend(fontsize=7,loc='lower right')
        vals=[latency('Dense',576)/latency('STRIDE',k) for k in (64,128,192)]
        axs[1].bar(['64','128','192'],vals,color='#087f8c',width=.55);axs[1].set(xlabel='Retained visual tokens',ylabel='End-to-end speedup',ylim=(0,2.05));axs[1].grid(axis='y',alpha=.2)
        for i,value in enumerate(vals):axs[1].text(i,value+.04,f'{value:.2f}x',ha='center')
        fig.savefig(out/'figures/quality_latency.pdf',metadata={'Creator':None,'Producer':None,'CreationDate':None});plt.close(fig)
    if args.artifact in {'intervals','all'}:
        import numpy as np
        from bootstrap import bootstrap_task
        confidence=[]
        for k in (64,128,192):
            comparator='VisPruner' if k==64 else 'Vision top-k control'
            distributions=[]
            for t in TASKS:
                sets=[read(ROOT/lookup[m,b,t]['source']) for m,b in [('STRIDE',k),(comparator,k),('Dense',576)]]
                if t=='mme':
                    allowed={(str(r['id']),r['question']) for r in mme_complete_pairs(sets[0])}
                    sets=[[r for r in rows if (str(r['id']),r['question']) in allowed] for rows in sets]
                ours,other,dense=bootstrap_task(t,sets);distributions.append((ours-other)/dense)
            ci=np.quantile(100*np.mean(distributions,axis=0),[.025,.975]).tolist()
            expected=next(r for r in json.loads((ROOT/'results/final/paired_intervals.json').read_text()) if r['budget']==k)
            assert np.allclose(ci,expected['ci95'],atol=1e-10,rtol=0)
            confidence.append({'budget':k,'comparator':comparator,'difference':quality('STRIDE',k)-quality(comparator,k),'ci95':ci})
        (out/'paired_intervals.json').write_text(json.dumps(confidence,indent=2)+'\n')
    if args.artifact=='all':
        import re
        def numbers(path):
            values=[]
            for line in path.read_text().splitlines():
                if ' & ' not in line:continue
                columns=line.replace(r'\\','').split('&')[1:]
                try: row=[float(re.sub(r'[^0-9.+-]','',v)) for v in columns]
                except ValueError:continue
                values.append(row)
            return values
        for name in ['main_results.tex','decoder_timing.tex','ablations.tex']:
            assert numbers(ROOT/'paper/tables'/name)==numbers(out/'tables'/name),name
    print(f"Verified {len(records)} main cells, {len(ablations)} ablation cells, 4 timing cells; artifact={args.artifact}")
if __name__=='__main__':main()
