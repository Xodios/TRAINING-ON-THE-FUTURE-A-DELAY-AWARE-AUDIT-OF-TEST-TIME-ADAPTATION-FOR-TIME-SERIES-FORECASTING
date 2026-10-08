# Integration kit (v8) for the reference TTA repository

Applies to a fresh clone of the reference repository (CC BY-NC-SA 4.0, obtained separately). We
distribute patches and a runner, never modified copies of the repository.

## apply_patches.py
Idempotent; run `python apply_patches.py` from the repository root. Three classes of changes
(paper Appendix A.2):
1. `LABEL_LAG`: delay-aware maturity for all four methods (TAFAS, COSA, PETSA, DynaTTA) and
   reduction of the causally observed partial-ground-truth prefix by the label lag l = d - H for
   all four, with guards when the prefix reaches zero. At l = 0 behavior is bit-identical to the
   authors' code.
2. `EXPORT_NPZ`: per-origin prediction export (`preds` of shape [T, C, H], origin-tagged) as the
   sole interface to our scoring pipeline.
3. Compatibility fixes for current library versions: the deprecated positional argument in
   `Series.apply` (datasets/build.py), and out-of-place non-stationary normalization
   (`x = x / stdev`) in models/forecast.py, models/PatchTST.py, models/normalize.py, plus a
   defensive patch of the same class in models/iTransformer.py. The in-place form breaks autograd
   for input-space methods (it crashed DynaTTA on the reference stack until patched).
`apply_patches_run1.log` is the output on a fresh clone; `apply_patches_run2.log` is the all-[skip]
second invocation demonstrating idempotency.

## run_external.py
`python run_external.py --method {frozen,tafas,cosa,petsa,dynatta} --dataset <ds> --data_dir <dir>
--seed <s> --delay <d> --out <file.npz> [--skip_train | --force_train] [--tag <suffix>]
[--opts KEY VALUE ...] [--spec <stream_spec.json>]`
Trains (or reuses) a per-(dataset, seed) checkpoint under results/ours_<ds>_PatchTST_s<seed>/, runs
the requested method at label-release delay d for any of the four methods, and writes the
origin-tagged predictions. `--tag` suffixes the exported method name (used for the released-script
configurations); `--opts` forwards yacs overrides; `--spec` asserts scaling alignment against a
harness stream specification.
