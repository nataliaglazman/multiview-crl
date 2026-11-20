"""Does a warm.py run's code carry each view's acquisition style: its gain, bias and noise level?

The renderer draws three style latents per view, independently for T1 and FLAIR (the generator's shared gain-bias
option is off). gain = 1 + 0.3 clamp(z0, -1, 1) scales every tissue and the lesion, bias = 0.1 clamp(z1, -1, 1) is
added to them, and the noise level is 0.01 + 0.05 |z2|. Banks don't store these latents, so they are redrawn per test
subject from the generator's own seeds, with the generator set up as bank.py sets it up. The redraw must reproduce the
bank's stored content latents. bank.py records none of its options, so the redraw tries the ones that change the draw
(--causal, --nuisance) and keeps the setting that reproduces them.

Per run, view and stage (code, pooled features, GAP), 5-fold R2 on the test subjects, by own_variation.py's ridge (the
features standardized and clipped, as the own-variation table reads content) and for the code also by gradient
boosting, of:

  own      the view's own gain, bias and noise level
  other    the other view's, drawn independently of this view: what own reads by chance, the null for own
  floor    both, for the run's untrained twin (TailCode's statistics filled from training subjects)

A code that carries no style reads its own view's style no better than the other view's. On a gain-off bank (bank.py
--gain off) the gain latent is drawn but never rendered, so gain is a null there too.

    python out/burden_feasibility_20261102/style_audit.py OUT_JSON RUN [RUN ...]

Only finished runs are read (checkpoint at the run's last step). Runs already in OUT_JSON at the same step are reused;
each bank's redrawn latents are cached beside OUT_JSON.
"""

import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import bank as bank_py
import code_audit
import numpy as np
import own_variation
import torch
import warm

import training.main_conv_synthetic as T

STYLE = ("gain", "bias", "noise")
VIEWS = ("t1", "flair")
STAGES = ("code", "code_boosted", "pooled", "gap")
SETTINGS = (  # the bank.py options that change the latent draw, most likely first
    {"causal": True, "nuisance": False},
    {"causal": False, "nuisance": False},
    {"causal": True, "nuisance": True},
    {"causal": False, "nuisance": True},
)


def style_targets(z_style):
    """Each subject's gain, bias and noise level as the renderer applies them, up to an affine map (R2 ignores it)."""
    z = np.asarray(z_style, dtype=np.float64)
    return np.column_stack([np.clip(z[:, 0], -1, 1), np.clip(z[:, 1], -1, 1), np.abs(z[:, 2])])


def generator(setting, total, split="test"):
    """A split's generator as bank.py builds it for a bank of ``total`` subjects in that split."""
    args = bank_py.run_args(0.4, nuisance=setting["nuisance"], causal=setting["causal"])
    inner = T.make_dataset(args, split, total)._inner
    if setting["causal"]:
        inner.scm = bank_py.causal_scm(args.seed)
    return inner


def draw(inner, i):
    """Subject ``i``'s content latents (as bank.py stores them) and both views' style latents."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a subject whose lesion doesn't fit is redrawn with a warning, as in bank.py
        d = inner._draw_pseudo_mri(inner.sample_seed_for(i))
    return (
        np.asarray(d["z_content"], dtype=np.float32)[:9],
        np.asarray(d["z_style_v1"], dtype=np.float64),
        np.asarray(d["z_style_v2"], dtype=np.float64),
    )


def draw_chunk(job):
    setting, total, start, stop, split = job
    torch.set_num_threads(1)
    inner = generator(setting, total, split)
    return start, [draw(inner, i) for i in range(start, stop)]


def matches(content, stored):
    return np.isclose(content, stored, rtol=0, atol=1e-6).all(axis=-1)


def redraw(root, n, workers=bank_py.WORKERS, check=4, split="test"):
    """The bank.py setting that reproduces ``root``'s stored content latents, and the first ``n`` ``split`` subjects'
    style latents (T1, FLAIR). Raises if no setting reproduces them, or if any redrawn subject differs."""
    stored = np.load(f"{root}/{split}_zc.npy")
    total = len(stored)
    for setting in SETTINGS:
        inner = generator(setting, total, split)
        if all(matches(draw(inner, i)[0], stored[i]) for i in range(min(check, n))):
            break
    else:
        raise ValueError(f"no bank.py setting tried {SETTINGS} reproduces the content latents stored in {root}")
    if workers <= 1:
        rows = [draw(inner, i) for i in range(n)]
    else:
        step = max(1, n // (workers * 2))
        jobs = [(setting, total, s, min(s + step, n), split) for s in range(0, n, step)]
        with ProcessPoolExecutor(workers) as pool:
            rows = [row for _, chunk in sorted(pool.map(draw_chunk, jobs), key=lambda c: c[0]) for row in chunk]
    bad = int((~matches(np.stack([r[0] for r in rows]), stored[:n])).sum())
    if bad:
        raise ValueError(f"redrawn content latents differ from those stored in {root} for {bad} of {n} subjects")
    return setting, [np.stack([r[1 + v] for r in rows]) for v in range(2)]


def bank_styles(root, n, cache):
    """``redraw``'s setting and both views' style targets, cached as ``cache``."""
    if cache.exists():
        saved = np.load(cache)
        return json.loads(str(saved["setting"])), [saved["t1"], saved["flair"]]
    setting, latents = redraw(root, n)
    styles = [style_targets(z) for z in latents]
    np.savez(cache, setting=json.dumps(setting), t1=styles[0], flair=styles[1])
    return setting, styles


def read(feats, styles):
    """Per view and stage, R2 of the view's own style targets and of the other view's."""
    out = {}
    for view, name in enumerate(VIEWS):
        out[name] = {}
        for stage in STAGES:
            source = "code" if stage == "code_boosted" else stage
            if source not in feats:
                continue
            x = feats[source][view]
            fit = code_audit.boosted if stage == "code_boosted" else own_variation.ridge
            out[name][stage] = {
                side: [fit(x, target[:, j]) for j in range(len(STYLE))]
                for side, target in (("own", styles[view]), ("other", styles[1 - view]))
            }
    return out


def twin_features(run, device):
    """The run's untrained twin's test features, as code_audit's floor builds them."""
    import causal_bundle  # it imports code_audit

    report, model, pool, residual, _ = code_audit.load_run(run, device, trained=False)
    a = report["args"]
    if pool is not None:
        causal_bundle.calibrate(model, pool, warm.Bank(a["bank"], "train", a["t1key"], residual=residual), device)
    bank = warm.Bank(a["bank"], "test", a["t1key"], int(a.get("test_subjects", 1000)), residual)
    return warm.features(model, bank, device, pool=pool)


def assess(run, device, out_json):
    report, model, pool, residual, step = code_audit.load_run(run, device)
    a = report["args"]
    bank = warm.Bank(a["bank"], "test", a["t1key"], int(a.get("test_subjects", 1000)), residual)
    n = len(bank.vox)
    cache = out_json.parent / f"style_{Path(a['bank']).name}_{n}.npz"
    setting, styles = bank_styles(a["bank"], n, cache)
    return {
        "step": step,
        "bank": a["bank"],
        "setting": setting,
        "read": read(warm.features(model, bank, device, pool=pool), styles),
        "floor": read(twin_features(run, device), styles),
    }


def show(name, row):
    options = " ".join(f"--{k}" for k, on in row["setting"].items() if on) or "no draw-changing options"
    print(
        f"{name}  step {row['step']}, bank {Path(row['bank']).name} (redrawn with bank.py {options}; content matches)"
    )
    print(f"    {'':5s} {'':12s} " + " ".join(f"{s:>14s}" for s in STYLE))
    for label, table in (("trained", row["read"]), ("untrained twin (the floor)", row["floor"])):
        print(f"  {label}:")
        for view in VIEWS:
            for stage in STAGES:
                if stage in table[view]:
                    cell = table[view][stage]
                    values = " ".join(f"{own:+6.2f} ({other:+5.2f})" for own, other in zip(cell["own"], cell["other"]))
                    print(f"    {view:5s} {stage:12s} {values}")


def main():
    out_json, runs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    device = warm.pick_device()
    results = json.loads(out_json.read_text()) if out_json.exists() else {}
    for run in runs:
        if not (run / "model.pt").exists():
            continue
        try:
            step = int(torch.load(run / "model.pt", map_location="cpu")["step"])
            last = int(json.loads((run / "report.json").read_text())["args"]["steps"])
        except Exception as error:  # a checkpoint being written right now
            print("skipping", run.name, "for now:", error, flush=True)
            continue
        if step != last or results.get(run.name, {}).get("step") == step:
            continue
        print("assessing", run.name, flush=True)
        results[run.name] = assess(run, device, out_json)
        out_json.write_text(json.dumps(results, indent=1))
    print(
        "== Style in each run's features: R2 of the view's own gain, bias and noise level, and in brackets of the other"
        " view's (drawn\n   independently: the null). 5-fold on test subjects; code_boosted by gradient boosting,"
        " the rest by ridge. On a gain-off\n   bank gain is never rendered, so its column is a null too."
    )
    for name in sorted(results):
        show(name, results[name])


if __name__ == "__main__":
    main()
