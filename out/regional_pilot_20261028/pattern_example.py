"""What removing K population patterns does to one subject's cell image, and what the patterns capture.

Prints each view's share of variance the K patterns explain at lesion-free and lesion-filled cells, and how well the
patterns' amounts read each generator factor on validation subjects (ridge fitted on train). Draws one validation subject,
picked by rule: the lesion volume closest to the median, slice through its fullest lesion cell.

    PYTHONPATH=SHIM:. python out/regional_pilot_20261028/pattern_example.py BANK_DIR OUT_PNG
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

sys.path.insert(0, "out/regional_pilot_20261028")
import residual_targets as RT  # noqa: E402

C, L, pilot = RT.C, RT.L, RT.pilot
K = 16
VIEWS = ("T1", "FLAIR")
LOW, MID, HIGH = "#2F6DB5", "#F0EFEC", "#C4472E"


def split(bank, view, model):
    """deviation from the population mean, the K-pattern fit of it, and the left-over, per subject on the 16^3 grid."""
    mean, cells, basis = model
    keep, y = bank.keep[:, 0], bank.targets[view][:, 0]
    deviation = ((y - mean) * keep)[:, cells].double()
    amounts = deviation @ basis[:K].T
    fit = amounts @ basis[:K]
    return deviation, fit, deviation - fit, amounts, cells, mean


def explained(deviation, fit, chosen):
    return float(1 - ((deviation - fit)[chosen] ** 2).sum() / (deviation[chosen] ** 2).sum())


def main():
    bank_dir, out_png = sys.argv[1], sys.argv[2]
    train, val = pilot.Bank(bank_dir, "train"), pilot.Bank(bank_dir, "val")
    from eval.metrics.dci import content_factor_names

    names = content_factor_names(9, "burden")
    lesion = torch.from_numpy(np.load(L.CACHE))
    models = [RT.population_basis(train, v) for v in range(2)]
    flat = lesion.reshape(len(lesion), -1)
    volume = flat.sum(1) * 64
    subject = int((volume - volume.median()).abs().argmin())
    cell = np.unravel_index(int(lesion[subject].argmax()), lesion.shape[1:])
    print(
        f"subject {subject}: lesion {volume[subject]:.0f} voxels, fullest cell {tuple(int(c) for c in cell)} ({100 * lesion[subject].max():.0f}% lesion)"
    )
    figure, axes = plt.subplots(2, 4, figsize=(10.5, 5.6), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("diverging", [LOW, MID, HIGH])
    for row, v in enumerate((1, 0)):
        view = VIEWS[v]
        deviation, fit, left, amounts, cells, mean = split(val, v, models[v])
        d_train, f_train, _, a_train, _, _ = split(train, v, models[v])
        keep = val.keep[:, 0][:, cells]
        share = lesion[:, cells]
        print(
            f"{view}: {K} patterns explain {100 * explained(d_train, f_train, train.keep[:, 0][:, cells]):.0f}% of train variance; "
            f"validation {100 * explained(deviation, fit, keep & (share == 0)):.0f}% at lesion-free cells, "
            f"{100 * explained(deviation, fit, keep & (share >= 0.5)):.0f}% at cells at least half lesion"
        )
        reads = {}
        for j, name in enumerate(names):
            if not name.startswith("unused"):
                ridge = Ridge(alpha=1.0).fit(a_train.numpy(), train.zc[:, j])
                reads[name[:12]] = round(float(ridge.score(amounts.numpy(), val.zc[:, j])), 2)
        print("  factors read from the 16 amounts:", reads)
        image = torch.full((4096,), float("nan")).reshape(16, 16, 16)
        panels = []
        for values in (
            deviation + mean[cells].double(),
            mean[cells].double().expand_as(deviation),
            fit + mean[cells].double(),
            left,
        ):
            grid = image.clone()
            grid[cells] = values[subject].float() if len(values) == len(deviation) else values
            grid[~val.keep[subject, 0]] = float("nan")
            panels.append(grid[:, :, cell[2]].numpy().T)
        low, high = np.nanpercentile(panels[0], [2, 98])
        limit = float(np.nanmax(np.abs(panels[3])))
        index = int(cells.flatten()[: cell[0] * 256 + cell[1] * 16 + cell[2]].sum())
        rank = int((left[subject].abs() > left[subject, index].abs()).sum()) + 1
        before = int((deviation[subject].abs() > deviation[subject, index].abs()).sum()) + 1
        print(
            f"  fullest lesion cell ranks {before} of {int(keep[subject].sum())} by size of deviation before removal, {rank} after"
        )
        titles = ("This subject", "Population average", f"Average + {K} patterns", "Left over")
        for column, (axis, panel, title) in enumerate(zip(axes[row], panels, titles)):
            shown = axis.imshow(
                panel,
                origin="lower",
                interpolation="nearest",
                cmap="gray" if column < 3 else cmap,
                vmin=low if column < 3 else -limit,
                vmax=high if column < 3 else limit,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_title(title, fontsize=12)
            if column == 0:
                axis.set_ylabel(view, fontsize=12)
            if column in (0, 3):
                axis.add_patch(
                    Rectangle((cell[0] - 0.5, cell[1] - 0.5), 1, 1, fill=False, edgecolor="#E8A317", linewidth=1.8)
                )
        figure.colorbar(shown, ax=axes[row, 3], fraction=0.046, pad=0.03, label="Brightness left over")
    figure.savefig(out_png, dpi=160)


if __name__ == "__main__":
    main()
