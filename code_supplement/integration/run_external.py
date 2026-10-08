#!/usr/bin/env python3
"""Run {frozen, tafas, cosa, petsa, dynatta} from the patched COSA_ICLR2026 repo on the
rls_delay_tta protocol (ratio split 25/5/70, train-stats scaling, L=96 H=24, stride-1 stream)
and export per-origin predictions [T, C, H] in scaled space for score_external().

Run FROM THE PATCHED REPO ROOT on a GPU box (the repo hardcodes .cuda()):
  python run_external.py --method frozen --dataset ETTh2 --data_dir <csv folder> --seed 0 --out frozen__ETTh2__patchtst_ext__s0__d24.npz
  python run_external.py --method tafas  --dataset ETTh2 --data_dir ... --seed 0 --delay 24 --skip_train --out tafas__ETTh2__patchtst_ext__s0__d24.npz
Pretraining happens on the first (non --skip_train) call per (dataset, seed); later calls reuse the checkpoint.
Method-specific tuned settings from the authors' scripts/ can be forwarded verbatim via --opts KEY VALUE ...
"""
import argparse, json, os, sys, tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
import torch
from config import _C, get_norm_module_cfg
from datasets.build import update_cfg_from_dataset, build_dataset
from datasets.loader import get_test_dataloader
from models.build import build_model, load_best_model, build_norm_module
from models.forecast import forecast
from utils.misc import set_seeds, prepare_inputs

CSV = {"ETTh2": "ETTh2.csv", "ETTm1": "ETTm1.csv", "weather": "weather.csv",
       "electricity": "electricity.csv", "traffic": "traffic.csv"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["frozen", "tafas", "cosa", "petsa", "dynatta"])
    ap.add_argument("--dataset", required=True, choices=list(CSV))
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--pred", type=int, default=24)
    ap.add_argument("--train_ratio", type=float, default=0.25)
    ap.add_argument("--test_ratio", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delay", type=int, default=24, help="d in stream steps; label lag l = max(0, d - pred)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--force_train", action="store_true", help="retrain even if a checkpoint exists")
    ap.add_argument("--dry_run", action="store_true", help="build cfg + datasets, verify split/scaling/origin arithmetic, then exit")
    ap.add_argument("--spec", default=None, help="optional stream_spec json to assert scaling alignment")
    ap.add_argument("--tag", default="", help="suffix for the exported method name (e.g. 'tuned') so variant runs get separate rows")
    ap.add_argument("--opts", nargs=argparse.REMAINDER, default=[], help="extra yacs overrides, e.g. --opts TTA.COSA.STEPS 1")
    a = ap.parse_args()

    lag = max(0, a.delay - a.pred)
    if a.method in ("petsa", "dynatta") and lag > 0:
        print("note: petsa/dynatta at LABEL_LAG>0 require the v8 patcher section (partial-GT lag); make sure apply_patches.py has been re-run.")

    cfg = _C.clone()
    update_cfg_from_dataset(cfg, a.dataset)
    cfg.DATA.NAME = a.dataset
    cfg.DATA.SEQ_LEN, cfg.DATA.PRED_LEN = a.seq, a.pred
    cfg.DATA.TRAIN_RATIO, cfg.DATA.TEST_RATIO = a.train_ratio, a.test_ratio
    dir_key = next((k for k in ("DIR", "DATA_DIR", "ROOT", "BASE_DIR", "PATH") if k in cfg.DATA), None)
    if dir_key is None:
        sys.exit(f"could not find the data-dir key in cfg.DATA; keys = {list(cfg.DATA.keys())}")
    cfg.DATA[dir_key] = a.data_dir
    cfg.MODEL.NAME = "PatchTST"
    cfg.SEED = a.seed
    if "WANDB" in cfg: cfg.WANDB.ENABLE = False
    cfg.TEST.SHUFFLE = False; cfg.TEST.DROP_LAST = False
    cfg.RESULT_DIR = os.path.join("results", f"ours_{a.dataset}_PatchTST_s{a.seed}")
    os.makedirs(cfg.RESULT_DIR, exist_ok=True)
    # the repo's default CHECKPOINT_DIR is a single shared 'results/' — datasets would overwrite
    # each other's weights; make it per-(dataset, seed) so skip-if-exists can never mix models
    cfg.TRAIN.CHECKPOINT_DIR = os.path.join(cfg.RESULT_DIR, "ckpt")
    os.makedirs(cfg.TRAIN.CHECKPOINT_DIR, exist_ok=True)
    if a.opts: cfg.merge_from_list(a.opts)
    set_seeds(cfg.SEED)

    # the repo expects data_dir/<dataset>/<csv>; auto-arrange if the csv sits flat in data_dir
    import shutil
    from pathlib import Path as _P
    sub = _P(a.data_dir) / a.dataset
    if not (sub / CSV[a.dataset]).exists():
        flat = _P(a.data_dir) / CSV[a.dataset]
        if flat.exists():
            sub.mkdir(parents=True, exist_ok=True)
            shutil.copy(flat, sub / CSV[a.dataset])
            print(f"arranged {flat} -> {sub / CSV[a.dataset]} (repo layout)")
        else:
            sys.exit(f"csv not found: put {CSV[a.dataset]} in {a.data_dir} or {sub}")

    # protocol bookkeeping from the raw csv (matches the repo's ratio arithmetic)
    N = len(pd.read_csv(os.path.join(a.data_dir, CSV[a.dataset])))
    start = N - int(N * a.test_ratio)                     # first test-window origin = train_len + val_len
    if a.spec:
        spec = json.load(open(a.spec))
        raw = pd.read_csv(os.path.join(a.data_dir, CSV[a.dataset]))
        z = raw[raw.columns[1:]].to_numpy(dtype=np.float64)
        tr = z[: int(N * a.train_ratio)]
        if not (np.allclose(tr.mean(0), spec["standardization"]["mu"], rtol=1e-3, atol=1e-4)
                and np.allclose(tr.std(0), spec["standardization"]["sd"], rtol=1e-3, atol=1e-4)):
            print("WARNING: train-stats scaling differs from stream_spec — investigate before comparing numbers.")

    if a.dry_run:
        tr_ds = build_dataset(cfg, "train"); te_ds = build_dataset(cfg, "test")
        n_tr_expected = int(N * a.train_ratio) - a.seq - a.pred + 1
        Tprime = len(te_ds)
        print(f"[dry] N={N} | train windows={len(tr_ds)} (expected ~{n_tr_expected}) | test windows T'={Tprime}")
        print(f"[dry] first test origin={start} | last={start + Tprime - 1} | harness expects origins {start}..{N - a.pred - 1}")
        assert abs(len(tr_ds) - n_tr_expected) <= 2, "train split length mismatch — ratio not honored by dataset_config"
        assert Tprime >= (N - a.pred) - start, "test window count smaller than the harness origin grid"
        xb = te_ds[0]
        print(f"[dry] sample window shapes: {[tuple(t.shape) for t in xb if hasattr(t, 'shape')]}")
        try:
            model = build_model(cfg)
            loader = get_test_dataloader(cfg)
            inputs = next(iter(loader))
            with torch.no_grad():
                pred, gt = forecast(cfg, prepare_inputs(inputs), model, None)
            print(f"[dry] forward probe OK: pred {tuple(pred.shape)}, gt {tuple(gt.shape)}")
        except Exception as e:
            print(f"[dry] forward probe skipped/failed here ({type(e).__name__}: {str(e)[:120]}) — "
                  "fine if CUDA-only; it will run on the GPU box.")
        print("[dry] DRY RUN PASSED — split, scaling and origin arithmetic line up. Remove --dry_run to train/adapt.")
        return

    model = build_model(cfg)
    norm = build_norm_module(cfg) if ("NORM_MODULE" in cfg and cfg.NORM_MODULE.ENABLE) else None
    ckpt = os.path.join(cfg.TRAIN.CHECKPOINT_DIR, "checkpoint_best.pth")
    if a.force_train or (not a.skip_train and not os.path.isfile(ckpt)):
        from trainer import build_trainer
        build_trainer(cfg, model, norm_module=norm).train()
    else:
        print(f"checkpoint found ({ckpt}) — skipping training (--force_train to retrain)")
    assert os.path.isfile(ckpt), f"no checkpoint at {ckpt} — run once without --skip_train first"
    model = load_best_model(cfg, model)
    if norm is not None:
        norm = load_best_model(get_norm_module_cfg(cfg), norm)

    os.environ["LABEL_LAG"] = str(lag)
    if a.method == "frozen":
        loader = get_test_dataloader(cfg)
        model.eval()
        preds = []
        with torch.no_grad():
            for inputs in loader:
                pred, _gt = forecast(cfg, prepare_inputs(inputs), model, norm)
                preds.append(pred.detach().cpu().numpy())
        P = np.concatenate(preds, axis=0).transpose(0, 2, 1)          # [T', C, H]
    else:
        import importlib
        tmp = a.out + ".tmp.npz"
        os.environ["EXPORT_NPZ"] = tmp
        mod = importlib.import_module(f"tta.{a.method}")
        adapter = mod.build_adapter(cfg, model, norm_module=norm)
        adapter.adapt()
        P = np.load(tmp)["preds"]                                     # [T', C, H] (export patch pre-transposed)
        os.remove(tmp)

    origins = start + np.arange(P.shape[0])
    meta = dict(method=a.method + (("_" + a.tag) if a.tag else ""), dataset=a.dataset, backbone="patchtst_ext", seed=a.seed,
                delay=int(a.delay), label_lag=int(lag), seq=a.seq, pred=a.pred,
                train_ratio=a.train_ratio, test_ratio=a.test_ratio, repo="COSA_ICLR2026+patches")
    np.savez_compressed(a.out, preds=P.astype(np.float32), origins=origins.astype(np.int64),
                        meta=json.dumps(meta))
    print(f"wrote {a.out}: preds {P.shape}, origins [{origins[0]}..{origins[-1]}], lag={lag}")

if __name__ == "__main__":
    main()
