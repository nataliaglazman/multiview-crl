"""mlmv brain renderer: the template's anatomy, the rendering's tissue physics, variants, eval and a CPU run."""

import csv
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch

from mlmv.config import load_config
from mlmv.synth.brain_template import REGION_NAMES, build_brain_template
from mlmv.synth.three_level import SyntheticSource, ThreeLevelGenerator
from mlmv.train.train_levels import run

ROOT = Path(__file__).resolve().parents[1]
SMALL = {
    "renderer": "brain",
    "image": [48, 56, 48],
    "lattice": [12, 14, 12],
    "lesion_radius": [0.5, 1.0],
    "brain": {"voxel_mm": 4.0},
}
_CACHE = {}


def gen(variant="exact", **kw):
    key = (variant, tuple(sorted((k, str(v)) for k, v in kw.items())))
    if key not in _CACHE:
        cfg = load_config(overrides={"synth": {**SMALL, "variant": variant, **kw}})
        _CACHE[key] = ThreeLevelGenerator(cfg.synth)
    return _CACHE[key]


class TestTemplate(unittest.TestCase):
    def test_anatomy(self):
        tpl = build_brain_template((96, 112, 96), 2.0)
        tissue, regions = tpl["tissue"], tpl["regions"]
        self.assertEqual(sorted(np.unique(tissue).tolist()), [0, 1, 2, 3, 4])
        self.assertEqual(sorted(np.unique(regions).tolist()), list(range(14)))
        self.assertTrue(((regions > 0) == (tissue > 0)).all())  # every brain voxel has a region
        brain_ml = (tissue > 0).sum() * 8 / 1000
        self.assertTrue(1000 < brain_ml < 1700, brain_ml)  # adult-like volume in millilitres
        for face in (tissue[0], tissue[-1], tissue[:, 0], tissue[:, -1], tissue[:, :, 0], tissue[:, :, -1]):
            self.assertFalse(face.any())  # the brain fits inside the grid
        x = np.arange(96)[:, None, None] - 47.5
        left = (regions >= 1) & (regions <= 6)
        right = (regions >= 7) & (regions <= 12)
        self.assertTrue((np.broadcast_to(x, regions.shape)[left] < 0).all())
        self.assertTrue((np.broadcast_to(x, regions.shape)[right] > 0).all())
        self.assertEqual(REGION_NAMES[3], "L temporal")

    def test_deterministic_and_scale_free(self):
        a, b = build_brain_template((48, 56, 48), 4.0), build_brain_template((48, 56, 48), 4.0)
        self.assertTrue(np.array_equal(a["tissue"], b["tissue"]))
        full = (build_brain_template((96, 112, 96), 2.0)["tissue"] > 0).sum() * 8
        half = (a["tissue"] > 0).sum() * 64
        self.assertLess(abs(full - half) / full, 0.1)  # same brain in millimetres at either resolution


class TestBrainRenderer(unittest.TestCase):
    def test_shapes_background_and_determinism(self):
        g = gen()
        lat = g.latents("test", [0, 1])
        t1, fl = g.render(lat)
        self.assertEqual(tuple(t1.shape), (2, 1, 48, 56, 48))
        outside = g.tissue_frac[0] == 1  # beyond the partial-volume edge
        self.assertGreater(int(outside.sum()), 0)
        self.assertEqual(float(t1[:, 0][:, outside].abs().max()), 0.0)  # skull-stripped: background is zero
        self.assertTrue(torch.equal(g.render(g.latents("test", [0, 1]))[1], fl))

    def test_batching_and_lesion_ground_truth(self):
        g = gen(lesion_blobs=[2, 3])
        together = g.latents("test", [4, 8])
        for j, i in enumerate([4, 8]):
            alone = g.latents("test", [i])
            for k in together:
                self.assertTrue(torch.allclose(together[k][j : j + 1].float(), alone[k].float(), atol=1e-6), k)
        # the lesion target is the cell average of the lesion actually drawn into the image
        vol = g._lesion_volume(together, 0)
        cells = torch.nn.functional.avg_pool3d(vol[None, None], g.m)[0]
        self.assertTrue(torch.allclose(together["lesion"][0], cells))
        self.assertGreater(float(together["lesion"][0].sum()), 0)

    def test_tissue_contrast(self):
        g = gen()
        lat = g.latents("test", range(4))
        t1, fl = g.render(lat)
        tissue = torch.from_numpy(g.template["tissue"])
        # majority-tissue voxels: at 4 mm the cortex has no voxel that is purely grey matter
        major = lambda t: (tissue == t) & (g.tissue_frac[t] > 0.5)  # noqa: E731
        mean = lambda img, t: float(img[:, 0][:, major(t)].mean())  # noqa: E731
        csf, gm, wm = 1, 2, 4
        self.assertTrue(mean(t1, wm) > mean(t1, gm) > mean(t1, csf))  # T1: WM bright, CSF dark
        self.assertTrue(mean(fl, gm) > mean(fl, wm) > mean(fl, csf))  # FLAIR: GM above WM, CSF suppressed

    def test_lesions_bright_in_flair_dark_in_t1(self):
        g = gen(lesion_blobs=[3, 3])
        lat = g.latents("test", [5])
        none = {**lat, "lesion_count": torch.zeros_like(lat["lesion_count"])}
        t1, fl = g.render(lat)
        t1_0, fl_0 = g.render(none)
        changed = (fl - fl_0).abs() > 1e-4
        self.assertTrue(changed.any())
        self.assertGreater(float((fl - fl_0)[changed].mean()), 0)
        self.assertLess(float((t1 - t1_0)[changed].mean()), 0)
        self.assertGreater(float(lat["lesion"].sum()), 0)
        self.assertLessEqual(float(lat["lesion"].max()), 1.0)

    def test_one_modality_lesion_and_shared_draws(self):
        g = gen("one_modality_lesion", lesion_blobs=[3, 3])
        lat = g.latents("test", [5])
        none = {**lat, "lesion_count": torch.zeros_like(lat["lesion_count"])}
        self.assertTrue(torch.equal(g.render(lat)[0], g.render(none)[0]))  # T1 never shows the lesion
        self.assertFalse(torch.equal(g.render(lat)[1], g.render(none)[1]))
        other = gen("noisy").latents("test", [5])
        for k in ("g", "r", "s", "w", "site", "style", "noise_seed"):
            self.assertTrue(torch.equal(gen().latents("test", [5])[k], other[k]), k)

    def test_site_and_field(self):
        g = gen()
        lat = g.latents("test", [2])
        a = g.render(lat)[0]
        b = g.render(lat, site=(lat["site"] + 1) % 6)[0]
        self.assertGreater(float((a - b).abs().mean()), 0)
        field = g.field(lat, "t1")[0, 4:8]
        self.assertEqual(float(field[:, g.regions == 0].abs().max()), 0.0)  # no regional factor outside the brain
        self.assertEqual(float(g.field(lat, "flair")[0, 15].abs().max()), 0.0)  # lesions are drawn, not in the field

    def test_source_geometry_inputs(self):
        src = SyntheticSource(load_config(overrides={"synth": SMALL}).synth, stride=2)
        self.assertEqual(src.feature_labels.shape, (24, 28, 24))
        self.assertTrue((src.feature_mask == (src.feature_labels > 0)).all())
        self.assertGreater(src.feature_mask.mean(), 0.1)
        self.assertLess(src.feature_mask.mean(), 0.6)


class TestBrainIntegration(unittest.TestCase):
    def test_cpu_run_with_raw_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(ROOT / "mlmv/config/runs/smoke_brain_cpu.yaml", overrides={"run": {"out_dir": tmp}})
            t0 = time.time()
            run_dir = run(cfg, log=lambda s: None)
            self.assertLess(time.time() - t0, 180)
            with open(run_dir / "eval" / "leakage.csv") as fp:
                rows = list(csv.DictReader(fp))
            self.assertEqual(len(rows), 2 * 3 * 6 + 6)  # modalities x blocks x factors, plus the image reference
            raw = [r for r in rows if r["block"] == "raw"]
            self.assertEqual({r["modality"] for r in raw}, {"t1+flair"})
            self.assertTrue(all(np.isfinite(float(r["best"])) for r in rows))


if __name__ == "__main__":
    unittest.main()
