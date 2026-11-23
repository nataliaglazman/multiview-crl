"""Matched, independent lesion-location and brain-size interventions."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

from eval.synthetic.synthetic_dataset import PseudoMRIRenderer

ARMS = ("A", "B", "Bp", "C", "D")
VIEWS = ("t1", "flair")
SPLITS = {"train": 1, "probe": 2, "test": 3, "pilot": 4}
VERSION = 2
# Brain size per view: fixed at 0, one draw shared by both views, or an independent draw per view.
# Bp ("B prime") has B's brain-size variation in every image, but its views disagree, so the loss can't use it.
BRAIN_RULE = {"A": "fixed", "B": "shared", "Bp": "per_view", "C": "fixed", "D": "shared"}
HIGH_CONTRAST_ARMS = ("C", "D")
COMMON_ARM = {"A": "A", "B": "A", "Bp": "A", "C": "C", "D": "C"}
# Input visibility must match within MATCHED_PAIRS and rise within INCREASED_PAIRS.
MATCHED_PAIRS = (("A", "B"), ("A", "Bp"), ("C", "D"))
INCREASED_PAIRS = (("A", "C"), ("B", "D"))
# Where lesion centres may sit. "wm": the original design, a complete sphere plus its blur halo inside the white
# matter shared by every brain size, so the competitor never changes the lesion's input evidence. "brain": any voxel
# of the smallest brain, with no margin, so the lesion can overlap grey matter, CSF, the fissure or the brain edge
# (parts outside the brain are masked) and brain size changes what surrounds it.
PLACEMENTS = ("wm", "brain")


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(safe_json(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class CompetitionRenderer:
    def __init__(self, res=64, radius=0.1, high_contrast=1.5, seed=42, brain_scale=0.5, placement="wm"):
        if placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of {PLACEMENTS}, got {placement!r}")
        self.res, self.radius = res, radius
        self.high_contrast, self.seed = high_contrast, seed
        self.brain_scale = brain_scale
        self.placement = placement
        # Feed already-squashed brain size; all other anatomical controls are zero.
        # The native lesion is discarded in favor of a common, explicit mask.
        self.renderer = PseudoMRIRenderer(res=res, lesion_radius=radius, content_squash="none")
        self.zero_deformation = torch.zeros(4, 4, 4)
        self.zero_fissure = torch.zeros(8, 8, 8)
        lower, upper = self.tissue(-brain_scale), self.tissue(brain_scale)
        if placement == "brain":
            # The smallest brain lies inside every larger one, so a centre in it is inside the brain at every size.
            admissible = lower.numpy() > 0
            for value in np.linspace(-brain_scale, brain_scale, 9):
                if np.any(admissible & (self.tissue(float(value)).numpy() == 0)):
                    raise RuntimeError("Brain nesting assumption failed")
            self.centres = np.argwhere(admissible)
            self.admissible = admissible
            return
        common = (lower.numpy() == 2) & (upper.numpy() == 2)
        distance = distance_transform_edt(np.pad(common, 1), sampling=2 / (res - 1))[1:-1, 1:-1, 1:-1]
        admissible = distance > radius + 1e-7
        # The complete blur neighbourhood of lesion voxels stays inside WM.
        # The outer response halo may see other tissues; its noise is audited.
        admissible = binary_erosion(admissible, structure=np.ones((3, 3, 3)))
        self.centres = np.argwhere(admissible)
        if len(self.centres) < 8:
            raise ValueError("Too few common lesion centres including blur halo; revise resolution/radius")
        self.admissible = admissible
        # With neutral other anatomy the WM sets are nested in brain size; the
        # smallest WM bound is therefore sufficient for every intermediate size.
        for value in np.linspace(-brain_scale, brain_scale, 9):
            if np.any(common & (self.tissue(float(value)).numpy() != 2)):
                raise RuntimeError("WM nesting assumption failed")

    def tissue(self, brain):
        z = torch.zeros(9)
        z[0] = brain
        return self.renderer.render_structure(z, self.zero_deformation, self.zero_fissure, "cpu", clean=True)[0]

    def subject(self, split, index):
        if split not in SPLITS or index < 0:
            raise ValueError("Invalid split/index")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, SPLITS[split], index]))
        raw_brain = float(rng.normal())
        centre = self.centres[rng.integers(len(self.centres))].copy()
        styles = rng.normal(size=(2, 3)).astype(np.float32)
        # Drawn last, so every earlier draw is unchanged; only Bp's FLAIR view uses it.
        raw_brain_other = float(rng.normal())
        # Seed namespaces are unique across subjects and splits, including noise redraws.
        namespace = (SPLITS[split] * 10_000_000 + index) * 10_000
        return dict(
            id=f"{split}:{index}",
            centre=centre,
            raw_brain=raw_brain,
            brain=float(self.brain_scale * np.tanh(raw_brain)),
            raw_brain_other=raw_brain_other,
            brain_other=float(self.brain_scale * np.tanh(raw_brain_other)),
            styles=styles,
            namespace=namespace,
        )

    @staticmethod
    def brain_sizes(subject, arm):
        """(T1, FLAIR) effective brain size in this arm."""
        rule = BRAIN_RULE[arm]
        if rule == "fixed":
            return 0.0, 0.0
        return subject["brain"], subject["brain"] if rule == "shared" else subject["brain_other"]

    def lesion(self, centre):
        coordinate = self.renderer.coords[tuple(np.asarray(centre, dtype=int))]
        return (torch.linalg.vector_norm(self.renderer.coords - coordinate, dim=-1) < self.radius).float()

    def render(self, subject, arm, norm=None, centre=None, noise_repeat=0, lesion_on=True):
        if arm not in ARMS or not 0 <= noise_repeat < 4000:
            raise ValueError("Invalid arm/noise repeat")
        contrast = self.high_contrast if arm in HIGH_CONTRAST_ARMS else 1.0
        lesion = self.lesion(subject["centre"] if centre is None else centre)
        tissues, images, masks = {}, [], []
        for v, (modality, brain) in enumerate(zip(("T1", "FLAIR"), self.brain_sizes(subject, arm))):
            if brain not in tissues:
                tissues[brain] = self.tissue(brain)
                if self.placement == "wm" and bool(((lesion > 0) & (tissues[brain] != 2)).any()):
                    raise RuntimeError("Lesion escaped common WM support")
            tissue = tissues[brain]
            mask = (tissue > 0).numpy()[None]
            raw = self.renderer.render_modality(
                tissue,
                lesion if lesion_on else torch.zeros_like(lesion),
                torch.from_numpy(subject["styles"][v]),
                modality,
                view_seed=subject["namespace"] + v,
                device="cpu",
                noise_seed=subject["namespace"] + 100 + 2 * noise_repeat + v,
                lesion_contrast_scale=contrast,
            ).numpy()
            if norm is not None:
                raw = (raw - norm["mean"]) / norm["scale"] * mask
            images.append(raw)
            masks.append(mask)
        return np.stack(images).astype(np.float32), np.stack(masks), lesion.numpy()

    def normalization(self, count):
        # Fit on B training subjects only, then save and reuse the affine everywhere.
        values = []
        for i in range(min(count, 64)):
            images, masks, _ = self.render(self.subject("train", i), "B")
            values.append(images[masks])
        values = np.concatenate([x.ravel() for x in values])
        if len(values) > 4_000_000:
            values = np.random.default_rng(0).choice(values, 4_000_000, replace=False)
        mean = float(values.mean())
        return dict(
            mean=mean,
            scale=max(float(np.quantile(np.abs(values - mean), 0.99)), 1e-6),
            source="B training subjects only",
            reference_subjects=min(count, 64),
        )


def prepare(args):
    root = Path(args.bank)
    if root.exists():
        raise FileExistsError(f"Bank already exists: {root}; use a new directory")
    r = CompetitionRenderer(
        args.res, args.radius, args.high_contrast, args.data_seed, args.brain_scale, args.lesion_placement
    )
    counts = dict(train=args.train_samples, probe=args.probe_samples, test=args.test_samples, pilot=args.pilot_samples)
    # Images are deterministic given these settings, so a bank of some arms (e.g. the arm-A
    # pilot) holds exactly the images those arms have in a bank of all five.
    arms = [a for a in ARMS if a in args.arms]
    norm = r.normalization(counts["train"])
    root.mkdir(parents=True)
    meta = dict(
        version=VERSION,
        ready=False,
        arms=arms,
        res=args.res,
        radius=args.radius,
        high_contrast=args.high_contrast,
        seed=args.data_seed,
        brain_scale=args.brain_scale,
        lesion_placement=args.lesion_placement,
        counts=counts,
        normalization=norm,
        centres_hash=array_hash(r.centres),
        centres_count=len(r.centres),
        files={},
    )
    np.save(root / "centres.npy", r.centres)
    write_json(root / "manifest.json", meta)
    gib = sum(counts[s] for s in ("train", "probe", "test")) * len(arms) * 2 * args.res**3 * 4 / 2**30
    print(f"Preparing {gib:.2f} GiB of float32 images for arms {', '.join(arms)} in {root}", flush=True)
    for split, count in counts.items():
        subjects = [r.subject(split, i) for i in range(count)]
        # Columns: lesion centre x/y/z (input voxels), brain size, and Bp's FLAIR brain size.
        labels = np.array([np.r_[s["centre"], s["brain"], s["brain_other"]] for s in subjects], dtype=np.float32)
        np.save(root / f"{split}_targets.npy", labels)
        write_json(root / f"{split}_subjects.json", subjects)
        if split == "pilot":
            continue
        for arm in arms:
            path = root / f"{split}_{arm}.npy"
            bank = np.lib.format.open_memmap(
                path, mode="w+", dtype="float32", shape=(count, 2, 1, args.res, args.res, args.res)
            )
            for i, s in enumerate(subjects):
                bank[i] = r.render(s, arm, norm)[0]
                if (i + 1) % 100 == 0:
                    print(f"{split}/{arm}: {i + 1}/{count}", flush=True)
            bank.flush()
            del bank
            meta["files"][path.name] = file_hash(path)
    for path in sorted(root.glob("*_targets.npy")) + sorted(root.glob("*_subjects.json")) + [root / "centres.npy"]:
        meta["files"][path.name] = file_hash(path)
    meta["ready"] = True
    meta["bank_id"] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    write_json(root / "manifest.json", meta)
    print(f"Bank ready: {root / 'manifest.json'}", flush=True)


def load_bank(root):
    root = Path(root)
    meta = json.loads((root / "manifest.json").read_text())
    if meta.get("version") != VERSION or not meta.get("ready"):
        raise ValueError("Incomplete or incompatible bank")
    return meta


def images(root, split, arm):
    return np.load(Path(root) / f"{split}_{arm}.npy", mmap_mode="r")


def targets(root, split):
    return np.load(Path(root) / f"{split}_targets.npy")


def bootstrap_interval(values, statistic=np.mean, repeats=1000):
    values = np.asarray(values)
    if len(values) < 2:
        return [None, None]
    rng = np.random.default_rng(714)
    draws = [statistic(values[rng.integers(len(values), size=len(values))]) for _ in range(repeats)]
    return np.quantile(draws, [0.025, 0.975]).tolist()


def audit(args):
    root = Path(args.bank)
    meta = load_bank(root)
    out = root / "audit"
    out.mkdir(exist_ok=args.overwrite)
    placement = meta.get("lesion_placement", "wm")
    r = CompetitionRenderer(
        meta["res"], meta["radius"], meta["high_contrast"], meta["seed"], meta["brain_scale"], placement
    )
    if array_hash(r.centres) != meta["centres_hash"]:
        raise ValueError("Renderer geometry differs from prepared bank")
    for filename, expected in meta["files"].items():
        if file_hash(root / filename) != expected:
            raise ValueError(f"Bank checksum mismatch: {filename}")
    # Only the arms in this bank are audited; a bank of arm A alone has no pairs to match, so its
    # gate reduces to geometry, reproducibility and axis coverage.
    arms = meta["arms"]
    matched_pairs = [p for p in MATCHED_PAIRS if set(p) <= set(arms)]
    increased_pairs = [p for p in INCREASED_PAIRS if set(p) <= set(arms)]
    for split in ("train", "probe", "test"):
        for arm in arms:
            cached = images(root, split, arm)
            for i in range(min(3, len(cached))):
                expected = r.render(r.subject(split, i), arm, meta["normalization"])[0]
                if not np.array_equal(expected, cached[i]):
                    raise ValueError("Current renderer does not reproduce the saved bank")
    subjects = [r.subject("pilot", i) for i in range(meta["counts"]["pilot"])]
    geometry = []
    # Under "brain" placement the competitor changes the tissue around the lesion by design, so the matched pairs'
    # lesion evidence is measured (how many subjects differ) instead of required to be identical.
    input_differs = {f"{a}_{b}": 0 for a, b in matched_pairs}
    for s in subjects:
        rendered = {a: r.render(s, a, meta["normalization"]) for a in arms}
        if len({array_hash(x[2]) for x in rendered.values()}) != 1:
            raise RuntimeError("Lesion masks differ between arms")
        lesion = rendered[arms[0]][2]
        roi = binary_dilation(lesion > 0, structure=np.ones((3, 3, 3)))
        off = {a: r.render(s, a, meta["normalization"], lesion_on=False)[0] for p in matched_pairs for a in p}
        core = lesion > 0
        for first, second in matched_pairs:
            response_a = (rendered[first][0] - off[first])[:, 0, roi]
            response_b = (rendered[second][0] - off[second])[:, 0, roi]
            same_core = np.array_equal(rendered[first][0][:, 0, core], rendered[second][0][:, 0, core])
            same_response = np.allclose(response_a, response_b, atol=5e-7, rtol=1e-5)
            if placement == "brain":
                input_differs[f"{first}_{second}"] += int(not (same_core and same_response))
                continue
            if not same_core:
                raise RuntimeError("Competitor changed lesion-core input evidence")
            if not same_response:
                raise RuntimeError("Competitor changed the rendered lesion intervention response")
        geometry.append(dict(subject=s["id"], mask_sha256=array_hash(lesion), voxels=float(lesion.sum())))
    # Save reviewable slices at the actual rendered lesion, with common scales per view.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(arms), figsize=(3 * len(arms), 6), squeeze=False)
    example = subjects[0]
    sample = [r.render(example, a, meta["normalization"])[0] for a in arms]
    iz = int(example["centre"][2])
    for v in range(2):
        lo = min(x[v, 0].min() for x in sample)
        hi = max(x[v, 0].max() for x in sample)
        for j, arm in enumerate(arms):
            axes[v, j].imshow(sample[j][v, 0, :, :, iz], cmap="gray", vmin=lo, vmax=hi)
            axes[v, j].set_title(f"{VIEWS[v]} / {arm}")
            axes[v, j].axis("off")
    fig.tight_layout()
    fig.savefig(out / "matched_images.png", dpi=150)
    plt.close(fig)
    rows, skipped = [], []
    # Direction/axis depend only on the subject index; no arm-specific pair selection.
    for i, s in enumerate(subjects[: args.noise_subjects]):
        axis = i % 3
        candidates = []
        for centre in r.centres:
            endpoint = centre.copy()
            endpoint[axis] += args.displacement
            if np.all(endpoint < r.res) and r.admissible[tuple(endpoint)]:
                candidates.append((centre, endpoint))
        if not candidates:
            skipped.append(dict(subject=s["id"], axis=axis))
            continue
        c1, c2 = candidates[np.random.default_rng([meta["seed"], i, 100]).integers(len(candidates))]
        roi = binary_dilation((r.lesion(c1).numpy() + r.lesion(c2).numpy()) > 0, structure=np.ones((3, 3, 3)))
        for arm in arms:
            endpoints = []
            for c in (c1, c2):
                endpoints.append(
                    np.stack(
                        [
                            r.render(s, arm, meta["normalization"], centre=c, noise_repeat=k + 1)[0][:, 0, roi]
                            for k in range(args.noise_repeats)
                        ]
                    ).astype(np.float64)
                )
            a, b = endpoints
            signal = np.sqrt(np.mean((b.mean(0) - a.mean(0)) ** 2, axis=1))
            noise = np.sqrt(0.5 * (a.var(0, ddof=1) + b.var(0, ddof=1)).mean(axis=1))
            for v, view in enumerate(VIEWS):
                rows.append(
                    dict(
                        subject=s["id"],
                        axis=axis,
                        arm=arm,
                        view=view,
                        signal=signal[v],
                        noise=noise[v],
                        visibility=signal[v] / noise[v],
                    )
                )
        print(f"Visibility noise audit: {i + 1}/{min(args.noise_subjects, len(subjects))}", flush=True)
    comparisons = {}
    for view in VIEWS:
        lookup = {(row["subject"], row["arm"]): row["visibility"] for row in rows if row["view"] == view}
        for a, b in matched_pairs + increased_pairs:
            values = np.array(
                [np.log(lookup[(s["id"], b)] / lookup[(s["id"], a)]) for s in subjects if (s["id"], a) in lookup]
            )
            ci = bootstrap_interval(values, np.median)
            passed = len(values) >= 2 and ci[0] >= np.log(0.9) and ci[1] <= np.log(1.1)
            comparisons[f"{view}_{b}_over_{a}"] = dict(
                median_ratio=float(np.exp(np.median(values))) if len(values) else None,
                ratio_ci=[None if x is None else float(np.exp(x)) for x in ci],
                equivalent_within_10_percent=bool(passed) if (a, b) in matched_pairs else None,
            )
    coverage = sorted({row["axis"] for row in rows})
    matched = all(
        comparisons[f"{view}_{b}_over_{a}"]["equivalent_within_10_percent"] for view in VIEWS for a, b in matched_pairs
    )
    if placement == "brain":
        # Brain size changes the lesion's surroundings by design here, so matched visibility is reported, not gated.
        matched = True
    increased = all(
        comparisons[f"{view}_{b}_over_{a}"]["ratio_ci"][0] is not None
        and comparisons[f"{view}_{b}_over_{a}"]["ratio_ci"][0] > 1
        for view in VIEWS
        for a, b in increased_pairs
    )
    # Per arm and view, so a faint view (e.g. T1) is visible even without a pair to compare.
    visibility = {}
    for arm in arms:
        for view in VIEWS:
            values = np.array([row["visibility"] for row in rows if row["arm"] == arm and row["view"] == view])
            visibility[f"{view}_{arm}"] = dict(
                median=float(np.median(values)) if len(values) else None,
                median_ci=bootstrap_interval(values, np.median),
                subjects=len(values),
            )
    result = dict(
        bank_id=meta["bank_id"],
        arms=arms,
        lesion_placement=placement,
        matched_pairs=matched_pairs,
        matched_pair_subjects_with_different_lesion_input=input_differs if placement == "brain" else None,
        increased_pairs=increased_pairs,
        geometry=geometry,
        noise_rows=rows,
        skipped=skipped,
        comparisons=comparisons,
        visibility=visibility,
        noise_repeats=args.noise_repeats,
        displacement=args.displacement,
        axis_coverage=coverage,
        input_gate_passed=bool(matched and increased and coverage == [0, 1, 2] and not skipped),
        scope="Input visibility only; supervised recovery and SSL comparison still required.",
    )
    write_json(out / "report.json", result)
    print(json.dumps(safe_json(dict(comparisons=comparisons, visibility=visibility)), indent=2), flush=True)
