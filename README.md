# Anonymous STRIDE reproduction package

This release supports the reported LLaVA-1.5-7B study only. Saved evidence covers six matched development subsets, native complete-pair MME scoring, POPE F1, routing/interface ablations, and synchronized decoder timing. It is not an independent confirmation of the paper's claims.

## Environment and setup
Historical runs record Python 3.10.9, PyTorch 2.10.0+cu128, CUDA runtime 12.8, transformers 4.51.3, 4-bit NF4, and bfloat16 computation on an RTX 4060 Laptop GPU. Install a compatible NVIDIA driver separately. The root requirements pin the current installed experiment environment; the complete historical transitive lock was not saved at run time.
```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest -p no:cacheprovider
```
All experiments use seed 0, greedy generation and max_new_tokens=12. Bootstrap uses 2,000 replicates and task seeds 20261011 + task index. GPU kernel and hardware differences can change answers and timings; exact fresh inference equivalence is not claimed.

## Saved-result reproduction (no model or dataset download)
```bash
python scripts/reproduce.py verify
python scripts/reproduce.py table1   # artifacts/tables/main_results.tex
python scripts/reproduce.py table2   # artifacts/tables/decoder_timing.tex
python scripts/reproduce.py table3   # artifacts/tables/ablations.tex
python scripts/reproduce.py figure1  # artifacts/figures/quality_latency.pdf
python scripts/reproduce.py intervals
(cd paper/figures && latexmk -pdf architecture.tex) # Figure 2
(cd paper && latexmk -pdf main.tex)
```
The table commands recompute scores from saved predictions, then validate against full-precision final CSV values. The rounded table numbers match the paper. Table formatting is compact rather than byte-identical to the manuscript. Figure 2 is a schematic, not a measured result. PDF metadata can be stripped with scripts/sanitize_media.py before sharing generated outputs.

## Public data and new inference
```bash
python scripts/prepare_paper_data.py
python scripts/run_matrix.py --config configs/main.yaml --dry-run
python scripts/run_matrix.py --config configs/main.yaml
python scripts/run_full_ablations.py --config configs/ablations.yaml
python scripts/rerun_additional.py semantic
python scripts/rerun_additional.py timing
```
Data are downloaded from public lmms-lab benchmark mirrors. No private Drive link, credential or project dataset is required. Check and comply with each upstream dataset/model license. Checkpoint: llava-hf/llava-1.5-7b-hf (public third-party account, not an author identity). Revisions were not frozen in the historical logs: the preparer stops if the upstream first 2,000 samples or decoded pixels differ from the frozen records. GQA must be materialized using the original full-split image pool; downloading only 2,000 images can change the selected questions. Preparation may download the full upstream splits into ignored work/, then materializes only the paper subset into ignored data_evaluation/. No benchmark images or model weights are included in this repository.

New inference writes runs/ separately from saved results/. Historical predictions remain the source of reported values. Raw end-to-end timings include preprocessing through generation; legacy preparation timing is not decoder prefill. Native MME scoring keeps 1,510 rows in complete opposite-target pairs from the 2,000 saved examples. The vision_topk control is the historical fastv alias, not official decoder-layer FastV.

## Scope and limitations
No NeXT/Qwen results, ShareGPT4V claims, sensitivity sweep, random-seed experiments or exploratory plots are included. The runtime retains shared algorithm primitives and compatibility aliases used by its tests; their presence is not evidence of an additional reported experiment. No assertion of universal semantic gain or statistically established aggregate superiority is made.
The source used by historical runs was a dirty checkout. Current source is provided, but exact historical implementation identity and independent GPU replication could not be certified. See REPRODUCIBILITY.md for tested and untested steps. Literature author names and required third-party license notices are attribution, not submission-author metadata; they are retained.
