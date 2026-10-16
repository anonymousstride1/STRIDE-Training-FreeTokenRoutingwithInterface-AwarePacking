# Verified scope

The current source is packaged with fixed router seed 0 and explicit Python, NumPy and PyTorch seed initialization for new runs. The historical manifests recorded seed 0, but did not certify deterministic GPU kernels. Fresh run seeds do not imply bitwise identical results across hardware or drivers.

The fresh smoke environment uses Python 3.10.9 and PyTorch 2.10.0+cpu. requirements-smoke.txt pins that environment; requirements.txt pins the currently installed CUDA experiment environment. CPU smoke testing is distinct from a full CUDA reproduction.

The following checks passed:
- 70 retained tests in a fresh, separate environment.
- Main matrix planner: 78 paper cells, seed 0, main backbone only, 4-bit weights, 2,000 examples and 12 generated tokens.
- Main quality scores recomputed from all 78 saved prediction cells.
- All 24 ablation-cell scores recomputed, paired example identity verified, and selected patch membership checked for order/position interventions.
- Four synchronized timing means recomputed from 100 predictions each.
- Semantic checks: 14 changed GQA strings, 57.50% without semantics; unchanged MME strings; 44.02% TextVQA without semantics.
- Three tables regenerated; every numeric row matches the manuscript tables exactly at their printed precision.
- Quality/latency figure regenerated from these values.
- The reported exploratory confidence interval endpoints regenerated from the frozen predictions with the original bootstrap seeds and 2,000 replicates.

## Unverified items

The full GPU inference matrix, fresh 4-bit environment installation, fresh model download, and upstream data redownload were not run during this release preparation. The historical dataset and model revisions were not saved, so reconstruction checks must stop if upstream sample order or pixels differ. Frozen records and pixel hashes permit detecting such changes; they cannot recover an unavailable upstream snapshot.

Historical runs used a dirty source checkout; the exact historical code state and complete run-time transitive environment cannot be independently certified. Current source edits are limited to release scoping, anonymous provenance, and explicit seed setup; reported saved predictions and metrics are unchanged.

Manuscript and architecture sources are provided with the official CVPR template and third-party notices preserved. Media metadata was removed by non-incremental PDF rewrite. Keep genuine license notices and literature citations. Figures are schematic or derived from saved results as described in the paper.

Content anonymization cannot prevent matching public method titles, code, or result fingerprints against an already public repository. Unknown coauthor/advisor identities cannot be scanned without any record of them. This release is clean against the discovered identity list, not a proof of unidentifiability.
