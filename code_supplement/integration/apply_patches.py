#!/usr/bin/env python3
"""Surgical patches for the COSA_ICLR2026 repo (which bundles TAFAS/COSA/PETSA/DynaTTA).
Adds (1) LABEL_LAG: extra label latency l = d - H read from env, shifting full-GT maturity
and shrinking the causally-observed partial-GT prefix; (2) EXPORT_NPZ: per-origin prediction
export in the scaled space. Idempotent; run from the repo root: python apply_patches.py
Every change is printed -> paste the log into the paper's deviations appendix."""
import sys
from pathlib import Path

def patch(path, old, new, required=True, allow_multi=False):
    p = Path(path); s = p.read_text()
    if new in s:
        print(f"  [skip] already applied in {path}"); return True
    if old not in s:
        msg = f"  [{'FAIL' if required else 'warn'}] anchor not found in {path}: {old[:60]!r}"
        print(msg)
        if required: sys.exit(1)
        return False
    s = s.replace(old, new) if allow_multi else s.replace(old, new, 1)
    p.write_text(s); print(f"  [ok] patched {path}"); return True

HDR = "import numpy as np"
LAG = ("import numpy as np\nimport os as _os\n"
       "LABEL_LAG = int(_os.environ.get('LABEL_LAG', '0'))   # extra label latency l = d - H (0 = native honest protocol)\n")
EXPORT_APPEND = ("""                if _os.environ.get('EXPORT_NPZ'):
                    self.__dict__.setdefault('_export_preds', []).append(pred.detach().cpu().numpy())
                self.mse_all.append(mse)""")
EXPORT_SAVE = ("""        assert len(self.mse_all) == len(self.test_loader.dataset)
        if _os.environ.get('EXPORT_NPZ'):
            _p = np.concatenate(self._export_preds, axis=0)          # [T', H, C], scaled space
            np.savez_compressed(_os.environ['EXPORT_NPZ'], preds=_p.transpose(0, 2, 1))
            print('exported predictions', _p.shape, '->', _os.environ['EXPORT_NPZ'])""")

for f, required in [("tta/tafas.py", True), ("tta/cosa.py", True), ("tta/petsa.py", False), ("tta/dynatta.py", False)]:
    print(f"--- {f} ---")
    ok = patch(f, HDR, LAG, required=required)
    if not ok: continue
    patch(f, "self.pred_step_end_dict[batch_idx] = self.cur_step + self.cfg.DATA.PRED_LEN",
             "self.pred_step_end_dict[batch_idx] = self.cur_step + self.cfg.DATA.PRED_LEN + LABEL_LAG",
             required=required, allow_multi=True)
    patch(f, "                self.mse_all.append(mse)", EXPORT_APPEND, required=required)
    patch(f, "        assert len(self.mse_all) == len(self.test_loader.dataset)", EXPORT_SAVE, required=required)

print("--- tta/tafas.py partial-GT lag ---")
patch("tta/tafas.py",
"""            pred_partial, ground_truth_partial = pred[0][:period], ground_truth[0][:period]

            loss_start = time.time()""",
"""            period_obs = max(0, period - LABEL_LAG)          # labels arrive l = d - H steps late
            if period_obs == 0:
                continue
            pred_partial, ground_truth_partial = pred[0][:period_obs], ground_truth[0][:period_obs]

            loss_start = time.time()""")
print("--- tta/cosa.py partial-GT lag ---")
patch("tta/cosa.py",
      "n_observed = min(batch_size - 1, self.cfg.DATA.PRED_LEN)",
      "n_observed = min(max(batch_size - 1 - LABEL_LAG, 0), self.cfg.DATA.PRED_LEN)")
patch("tta/cosa.py",
      "obs_j = min(max(batch_size - 1 - j, 0), pred_len)",
      "obs_j = min(max(batch_size - 1 - j - LABEL_LAG, 0), pred_len)")

print("--- datasets/build.py pandas>=2 compat (deprecated positional arg in Series.apply) ---")
for expr in ("row.month", "row.day", "row.weekday()", "row.hour", "row.minute"):
    patch("datasets/build.py",
          f"df_stamp.date.apply(lambda row: {expr}, 1)",
          f"df_stamp.date.apply(lambda row: {expr})", required=False, allow_multi=True)

print("--- autograd compat: out-of-place NST normalization (in-place div breaks input-graph TTA, e.g. DynaTTA) ---")
patch("models/forecast.py", "        enc_window /= stdev", "        enc_window = enc_window / stdev", required=True)
patch("models/normalize.py", "        enc_window /= stdev", "        enc_window = enc_window / stdev", required=False)
patch("models/PatchTST.py", "        x_enc /= stdev", "        x_enc = x_enc / stdev", required=True, allow_multi=True)
patch("models/iTransformer.py", "        x_enc /= stdev", "        x_enc = x_enc / stdev", required=False, allow_multi=True)

def insert_after(path, anchor, new_lines, required=True):
    """Insert new_lines (list of str, already indented) after the unique line containing `anchor`."""
    p = Path(path); s = p.read_text()
    if new_lines[0].strip() in s:                      # key on the first (marker-bearing) line only
        print(f"  [skip] already applied in {path}"); return True
    lines = s.splitlines(keepends=True)
    hits = [i for i, l in enumerate(lines) if anchor in l]
    if len(hits) != 1:
        print(f"  [{'FAIL' if required else 'warn'}] anchor not unique/found in {path}: {anchor[:60]!r} ({len(hits)} hits)")
        if required: sys.exit(1)
        return False
    i = hits[0]
    lines[i:i+1] = [lines[i]] + [nl if nl.endswith("\n") else nl + "\n" for nl in new_lines]
    p.write_text("".join(lines)); print(f"  [ok] patched {path}"); return True

print("--- PETSA/DynaTTA partial-GT lag (v8: puts both methods on the delay axis) ---")
# PETSA: reduce the causal prefix by the label lag; skip the update when nothing is observed yet
insert_after("tta/petsa.py", '"""Partial ground truth adaptation with time measurement"""',
             ["        period = max(0, period - LABEL_LAG)          # labels arrive l = d - H steps late"])
insert_after("tta/petsa.py", "self.time_counts['partial_forward_pass'] += 1",
             ["        if period == 0:                                   # nothing causally observed yet under LABEL_LAG",
              "            self.switch_model_to_eval()",
              "            return pred, ground_truth"])
# DynaTTA: its maturity dict is named pred_end (hence the earlier [warn]); shift it, reduce the prefix, guard the update
patch("tta/dynatta.py", "self.pred_end[batch_idx] = self.cur_step + self.cfg.DATA.PRED_LEN",
      "self.pred_end[batch_idx] = self.cur_step + self.cfg.DATA.PRED_LEN + LABEL_LAG")
insert_after("tta/dynatta.py", "def _adapt_partial(self, window, period, batch_size, batch_idx):",
             ["        period = max(0, period - LABEL_LAG)          # labels arrive l = d - H steps late"])
patch("tta/dynatta.py", "                self._update_rtab_partial(window, period, batch_idx)",
      "                if period > 0: self._update_rtab_partial(window, period, batch_idx)")
patch("tta/dynatta.py",
"""                pred_p, gt_p = pred[0][:period].clone(), gt[0][:period].clone()
                loss = F.mse_loss(pred_p, gt_p)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()""",
"""                if period > 0:                                 # skip the update when nothing is causally observed
                    pred_p, gt_p = pred[0][:period].clone(), gt[0][:period].clone()
                    loss = F.mse_loss(pred_p, gt_p)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()""")

print("NOTE: with the v8 section applied, PETSA and DynaTTA can be run at LABEL_LAG>0 as well.")
print("done.")
