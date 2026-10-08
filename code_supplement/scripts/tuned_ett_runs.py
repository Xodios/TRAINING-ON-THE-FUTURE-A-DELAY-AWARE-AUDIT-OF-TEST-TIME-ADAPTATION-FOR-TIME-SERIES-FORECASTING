"""Released-script (Appendix F) configurations on ETT. Run after external_extension_runs Steps 0-1."""
import os, subprocess, glob
BASE = os.environ.get("RLS_TTA_BASE", os.path.expanduser("~/ICLR"))
REPO, PY = f"{BASE}/COSA_ICLR2026", os.path.expanduser("~/tta_venv/bin/python")
OUT, D = f"{BASE}/rls_delay_tta/external_preds", f"{BASE}/rls_delay_tta/data"
TAFAS = ["TTA.SOLVER.BASE_LR","0.001","TTA.SOLVER.WEIGHT_DECAY","0.0001","TTA.TAFAS.GATING_INIT","0.01"]
COSA  = ["TTA.SOLVER.BASE_LR","0.001","TTA.SOLVER.WEIGHT_DECAY","0.0001","TTA.COSA.BATCH_SIZE","48",
         "TTA.COSA.STEPS","3","TTA.COSA.BUFFER_CONTEXT_SIZE","10","TTA.COSA.ADAPTER_TYPE","Linear",
         "TTA.COSA.FAST_ADAPTATION","True","TTA.COSA.PER_BATCH_LR_RESET","True","TTA.COSA.ADAPTIVE_LR","True",
         "TTA.COSA.PAAS","True","TTA.COSA.PERIOD_N","1","TTA.COSA.POGT","True",
         "TTA.COSA.PARTIAL_ADAPT","True","TTA.COSA.ADJUST_PRED","True"]
for M, OPTS in [("tafas", TAFAS), ("cosa", COSA)]:
    for DS in ["ETTh2", "ETTm1"]:
        for DEL in [24, 48, 96]:
            out = f"{OUT}/{M}_tuned__{DS}__patchtst_ext__s0__d{DEL}.npz"
            if os.path.exists(out) and os.path.getsize(out) > 100_000: print("have", os.path.basename(out)); continue
            subprocess.run([PY, "run_external.py", "--method", M, "--dataset", DS, "--data_dir", D, "--seed", "0",
                            "--delay", str(DEL), "--skip_train", "--tag", "tuned", "--out", out, "--opts"] + OPTS, cwd=REPO)
print("tuned files:", len(glob.glob(f"{OUT}/*_tuned__*.npz")))
