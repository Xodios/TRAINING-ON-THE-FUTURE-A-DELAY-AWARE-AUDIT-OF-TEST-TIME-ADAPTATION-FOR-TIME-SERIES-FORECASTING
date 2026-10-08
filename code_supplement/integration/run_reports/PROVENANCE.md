# Run reports
Per-run JSON blocks emitted by the reference repository's own instrumentation, all at label lag
l = d - H (d = 24 means l = 0). Contents of this folder:
- `tafas__*`, `cosa__*`, `petsa__*` (12 files): the original external batch on the authors'
  checkpoints. The COSA files carry its own `partial_gt_adaptation_count`: 487 (ETTh2, d=24) and
  1,951 (ETTm1, d=24), and exactly 0 at d=48/96, the counters behind Figure 4 (right panel).
- `dynatta__ETTh2__s0__d24.json`, `dynatta__ETTm1__s0__d24.json`: the original DynaTTA runs
  (test MSE 1.3821 / 0.6188) after the out-of-place normalization fix first made them runnable.
- `dynatta__*__reverification__*.json`: DynaTTA re-run for verification (console capture);
  reproduces the original MSEs within 1.1% / 0.05% and confirms 11,196 trainable parameters and
  972 / 3,900 adaptation events.
- `tuned_runs_summary.json`: released-script (Appendix F) runs; fields extracted verbatim from the
  runs' own reports, including the partial-ground-truth counters under lag (79/60 on ETTh2,
  508/496 on ETTm1) and the 5,887-parameter adapter.
DynaTTA-on-Traffic was not run (paper Limitations); no report exists for it.
