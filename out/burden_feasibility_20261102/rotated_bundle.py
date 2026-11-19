"""A bundle's codes rotated into their cross-view canonical directions, written as a bundle of its own, so label-free
discovery (eval.causal.latent_causal_discovery) runs on those directions instead of the code's units. No labels: the
rotation uses the two views' codes only.

    python out/burden_feasibility_20261102/rotated_bundle.py IN.npz OUT.npz [--components K | --all-pairs]

Each view's code is standardized, CCA between the views gives each view a weight matrix, and each view's first K
canonical variates replace its code. K is the number of canonical pairs that agree beyond a shuffled-pairing null (the
count the tabular benchmark's proximal arms use) unless --components sets it; --all-pairs keeps every pair, a pure
rotation of the code (an untrained code keeps few pairs beyond chance, fewer than the factors PC names). Canonical directions are unique when
their correlations differ, so factors the views share at different strengths come out separately; directions with
tied correlations stay mixed. The factors, their own noise, the graph and the identity record are copied unchanged, so
the rotated bundle describes the same rows. The rotation is fitted on the bundle's own rows, which is label-free.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from eval.causal.effect_estimation import _matrix, canonical_pairs, shared_directions


def rotate(view1, view2, components=None):
    """Each view's first K canonical variates (n, K), then K and every canonical correlation, largest first."""
    a, b = _matrix(view1, len(view1), "view 1 code"), _matrix(view2, len(view1), "view 2 code")
    left, right, correlations = canonical_pairs(a, b)
    k = shared_directions(a, b) if components is None else components
    if not 1 <= k <= len(correlations):
        raise ValueError(f"need between 1 and {len(correlations)} components, got {k}")
    return a @ left[:, :k], b @ right[:, :k], int(k), correlations


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bundle", type=Path)
    p.add_argument("out", type=Path)
    keep = p.add_mutually_exclusive_group()
    keep.add_argument("--components", type=int, default=None, help="canonical pairs to keep (default: beyond chance)")
    keep.add_argument("--all-pairs", action="store_true", help="keep every canonical pair: a pure rotation")
    a = p.parse_args()
    data = np.load(a.bundle, allow_pickle=False)
    components = min(data["emb_view1"].shape[1], data["emb_view2"].shape[1]) if a.all_pairs else a.components
    view1, view2, k, correlations = rotate(data["emb_view1"], data["emb_view2"], components)
    meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
    meta.update(
        rotation="cca",
        components=k,
        canonical_r=[round(float(r), 4) for r in correlations],
        source_bundle=str(a.bundle),
    )
    arrays = {key: data[key] for key in data.files if key not in ("emb_view1", "emb_view2", "meta")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out,
        **arrays,
        emb_view1=view1.astype(np.float32),
        emb_view2=view2.astype(np.float32),
        meta=json.dumps(meta, indent=2),
    )
    print(f"{a.out}: {k} of {len(correlations)} canonical pairs, r {' '.join(f'{r:.2f}' for r in correlations)}")


if __name__ == "__main__":
    main()
