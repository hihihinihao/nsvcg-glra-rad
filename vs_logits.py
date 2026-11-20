"""Fit per-class vector scaling (VS) on val logits and report test raw/VS.

Usage:
  python vs_logits.py --run runs/mnre_glra_front_rad_seed42
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score


def fit_vs(z, y, lam=1e-3):
    z = z.astype(np.float64)
    n, c = z.shape

    def obj(p):
        a, b = p[:c], p[c:]
        q = z * a + b
        q = q - q.max(1, keepdims=True)
        logp = q - np.log(np.exp(q).sum(1, keepdims=True))
        return -float(logp[np.arange(n), y].mean()) + lam * (
            float(((a - 1) ** 2).sum()) + float((b ** 2).sum()))

    r = minimize(obj, np.concatenate([np.ones(c), np.zeros(c)]),
                 method="L-BFGS-B", options={"maxiter": 2000})
    return r.x[:c], r.x[c:]


def metrics(y, z):
    p = z.argmax(1)
    return {
        "micro_f1": 100 * f1_score(y, p, labels=range(1, z.shape[1]), average="micro", zero_division=0),
        "macro_f1_relations": 100 * f1_score(y, p, labels=range(1, z.shape[1]), average="macro", zero_division=0),
        "accuracy": 100 * float((p == y).mean()),
        "none_prediction_ratio": 100 * float((p == 0).mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--lam", type=float, default=1e-3)
    args = p.parse_args()
    run = Path(args.run)
    va = np.load(run / "val_logits.npz")
    te = np.load(run / "test_logits.npz")
    a, b = fit_vs(va["raw"], va["gold"], args.lam)
    raw = metrics(te["gold"], te["raw"])
    vs = metrics(te["gold"], te["raw"] * a + b)
    print("raw", json.dumps(raw, indent=2))
    print("VS ", json.dumps(vs, indent=2))
    (run / "vs.json").write_text(json.dumps({"raw": raw, "vs": vs, "lam": args.lam},
                                            indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
