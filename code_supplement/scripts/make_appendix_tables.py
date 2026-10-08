#!/usr/bin/env python3
"""Build Appendix E (all CIs) and Appendix F (default vs released-script config) LaTeX tables
from the on-disk result files. Usage: python make_appendix_tables.py <base>/rls_delay_tta
Writes appendix_E_F_tables.tex into that folder. Requires \usepackage{booktabs,longtable}."""
import json, sys, glob
from pathlib import Path
import pandas as pd

_argv = [a for a in sys.argv[1:] if not a.startswith("-")]          # ignore kernel args in notebooks
ROOT = Path(_argv[0]) if _argv else Path.home() / "ICLR" / "rls_delay_tta"
out = []

def esc(s): return str(s).replace("_", r"\_")

# ---------------- Appendix F: default vs released-script configurations ----------------
vals = {}
for ds in ["ETTh2", "ETTm1"]:
    T = pd.read_csv(ROOT / "results" / f"table_{ds}_patchtst_ext.csv")
    mcol = "method" if "method" in T.columns else T.columns[1]
    T = T.drop_duplicates(subset=[mcol], keep="last").set_index(mcol)
    dcols = sorted([c for c in T.columns if c.replace("d=", "").strip().isdigit()],
                   key=lambda c: int(c.replace("d=", "")))
    dcols = [c for c in dcols if int(c.replace("d=", "")) >= 24]
    vals[ds] = {m: [T.loc[m, c] for c in dcols] for m in
                ["frozen", "tafas", "tafas_tuned", "cosa", "cosa_tuned", "rls_bank_med", "naive"] if m in T.index}

out += [r"% ===== Appendix F table (auto-generated from table_*_patchtst_ext.csv) =====",
        r"\begin{table}[h]\centering\scriptsize",
        r"\caption{Repository-default vs.\ released-script configurations on the authors' own",
        r"checkpoints (standardized MSE, origin-aligned). The script configuration is significantly",
        r"worse on ETTm1 at every delay and frozen-equivalent on ETTh2 ($|\Delta|\le 0.0005$, n.s.);",
        r"no single released configuration is defensible on both streams. Script-configuration COSA",
        r"uses a larger context buffer (5{,}887 parameters vs.\ 5{,}047).}",
        r"\begin{tabular}{llrrr}\toprule",
        r" & configuration & $d{=}H$ & $d{=}2H$ & $d{=}4H$ \\\midrule"]
for ds in ["ETTh2", "ETTm1"]:
    v = vals[ds]
    def row(label, m): return f" & {label} & " + " & ".join(f"{x:.4f}" for x in v[m]) + r" \\"
    out += [f"{ds}" + row("TAFAS default", "tafas"), row("TAFAS script", "tafas_tuned"),
            row("COSA default", "cosa"), row("COSA script", "cosa_tuned"),
            row(r"RLS-Bank (ref.)", "rls_bank_med"),
            f" & \\emph{{frozen {v['frozen'][0]:.4f}; persistence {v['naive'][0]:.4f}}} & & & \\\\",
            r"\addlinespace"]
out += [r"\bottomrule\end{tabular}\end{table}", ""]

# ---------------- Appendix E: all bootstrap CIs (deduplicated, longtable-ready rows) ----------------
def ci_table(path, caption, cols):
    rows = json.loads((ROOT / path).read_text())
    seen, dd = set(), []
    for r in rows:
        k = (r.get("dataset"), r.get("backbone"), r.get("seed"), r.get("delay"), r.get("pair"))
        if k not in seen: seen.add(k); dd.append(r)
    rows = dd
    body = [r"\begin{table}[h]\centering\tiny", rf"\caption{{{caption}}}",
            r"\begin{tabular}{" + "l" * (len(cols) - 3) + r"rrr}\toprule",
            " & ".join(esc(c) for c in cols) + r" \\\midrule"]
    for r in rows:
        star = r"$^{*}$" if r["significant"] else ""
        cells = [f"{r[c]:.4f}" if isinstance(r[c], float) else esc(r.get(c, "")) for c in cols]
        cells[cols.index("pair")] = esc(r["pair"]) + star
        body.append(" & ".join(str(c) for c in cells) + r" \\")
    body += [r"\bottomrule\end{tabular}\end{table}", ""]
    return len(rows), body

n_int, b1 = ci_table("bootstrap_ci.json",
    r"Internal cells: paired per-step moving-block bootstrap (block $=2\times$ period, $B{=}500$). "
    r"Negative mean $=$ first method better; $^{*}$ $=$ 95\% CI excludes zero.",
    ["dataset", "backbone", "delay", "pair", "mean_diff", "ci_lo", "ci_hi"])
n_ext, b2 = ci_table("bootstrap_ci_external.json",
    r"External comparisons on the authors' own checkpoints (origin-aligned streams; same bootstrap), "
    r"including the released-script (\texttt{\_tuned}) configurations and per-seed rows.",
    ["dataset", "seed", "delay", "pair", "mean_diff", "ci_lo", "ci_hi"])
out += [r"% ===== Appendix E tables (auto-generated from bootstrap_ci*.json) ====="] + b1 + b2

(ROOT / "appendix_E_F_tables.tex").write_text("\n".join(out))
print(f"wrote {ROOT / 'appendix_E_F_tables.tex'} ({n_int} internal + {n_ext} external CI rows)")
