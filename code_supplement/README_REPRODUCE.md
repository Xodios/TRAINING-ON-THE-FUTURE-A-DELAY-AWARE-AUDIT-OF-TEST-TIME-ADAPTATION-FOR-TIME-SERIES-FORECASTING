# Reproducibility package: Training on the Future (delay-aware TTA audit)

Anonymous supplementary code (MIT, see LICENSE). No datasets, checkpoints or machine logs are
included; everything regenerates from public sources. `results_cache/` holds the cached result
tables, the bootstrap CI records, and the exact run manifest so every number in the paper can be
checked without recomputation. Tested with Python 3.12 and CUDA torch 2.12 (Appendix A.1).

## Layout
notebooks/    main harness (internal grid, ELF-style corrector, ablations, ingestion, bootstrap
              CIs, all paper figures), an alternative-split edition of the harness (70/10/20; not
              used in the paper), and the external-run orchestrator
integration/  v8 patcher + runner for the reference TTA repository, the two patcher logs,
              stream_specs/ (per-stream scaling specifications), and run_reports/ (the runs' own
              instrumentation JSON backing the counter and re-verification claims)
scripts/      appendix-table generator and the released-script (Appendix F) configuration runs
results_cache/ table_*.csv, headline_*.csv, bootstrap_ci*.json, config_snapshot.json,
              run_manifest.csv (one row per run backing the paper: 2144 internal-grid and 904 external-replay runs on full-length streams; development runs on truncated or pre-alignment streams are excluded)
LICENSE       MIT (this code only; the reference repository is CC BY-NC-SA 4.0, obtained separately)

## Steps
1. Set `RLS_TTA_BASE` (default `~/ICLR`) and place this folder there. Data auto-fetches on first
   run (ETT from GitHub; Weather/Electricity/Traffic from the public Autoformer dataset folder,
   Google Drive id 1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) into `rls_delay_tta/data/`.
2. `python3 -m venv --system-site-packages ~/tta_venv && ~/tta_venv/bin/pip install -r requirements.txt`
3. Run All on `notebooks/rls_delay_tta_full_run.ipynb` (internal grid; 10-14 h CPU first pass,
   cached and safely resumable; the coded training-batch reductions on the wide streams are
   described in Section 3 of the paper).
4. Clone the reference repository next to it as `COSA_ICLR2026/`, copy `integration/apply_patches.py`
   and `integration/run_external.py` in, run `python apply_patches.py` (expect the run1 log), then
   run `notebooks/external_extension_runs.ipynb` Steps 0-3 (GPU; ~40 GB suffices) and
   `python scripts/tuned_ett_runs.py`.
5. Run All on the main notebook again (ingests external predictions, computes all external CIs,
   regenerates every paper figure into `rls_delay_tta/figures/paper/`).
6. `python scripts/make_appendix_tables.py <base>/rls_delay_tta` regenerates the appendix LaTeX tables.

Determinism: the internal pipeline is deterministic given a backbone (BLAS-level, verified across
machines); deep backbones use fixed seeds 0-2. GPU nondeterminism means external MSEs reproduce to
~1 percent, matching the paper's reported re-verification check.
