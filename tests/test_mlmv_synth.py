"""mlmv three-level generator: determinism, shared draws across variants, and what each variant changes."""

import unittest

import numpy as np
import torch

from mlmv.config import load_config
from mlmv.synth.three_level import ThreeLevelGenerator

SMALL = {"image": 32, "lattice": 8, "decoder_hidden": 32}


def gen(variant="exact", **kw):
    cfg = load_config(overrides={"synth": {**SMALL, "variant": variant, **kw}})
    return ThreeLevelGenerator(cfg.synth)


class TestGenerator(unittest.TestCase):
    def test_shapes_and_determinism(self):
        g = gen()
        lat = g.latents("train", [3, 4])
        t1, fl = g.render(lat)
        self.assertEqual(tuple(t1.shape), (2, 1, 32, 32, 32))
        self.assertEqual(tuple(lat["r"].shape), (2, g.n_regions, 4))
        self.assertEqual(tuple(lat["s"].shape), (2, 4, 8, 8, 8))
        again = g.latents("train", [3, 4])
        for k in lat:
            self.assertTrue(torch.equal(lat[k], again[k]), k)
        self.assertFalse(torch.equal(g.latents("test", [3])["g"], lat["g"][:1]))
        self.assertTrue(torch.equal(g.render(again)[0], t1))

    def test_variants_share_draws(self):
        a, b = gen("exact").latents("train", [7]), gen("noisy").latents("train", [7])
        for k in ("g", "r", "s", "w", "site", "style", "lesion"):
            self.assertTrue(torch.equal(a[k], b[k]), k)

    def test_regional_factor_fills_its_region(self):
        g = gen()
        lat = g.latents("train", [0])
        field = g.field(lat, "t1")[0, 4:8]  # r channels
        for region in range(1, g.n_regions + 1):
            cells = field[:, g.regions == region]
            self.assertTrue(torch.allclose(cells, lat["r"][0, region - 1][:, None].expand_as(cells)))

    def test_noisy_adds_eta(self):
        g = gen("noisy", noise_tau2=0.5)
        lat = g.latents("train", [0])
        field = g.field(lat, "t1")[0]
        expected_g = lat["g"][0][:, None, None, None] + np.sqrt(0.5) * lat["eta_g"][0]
        self.assertTrue(torch.allclose(field[:4], expected_g, atol=1e-6))

    def test_one_modality_lesion(self):
        g = gen("one_modality_lesion", lesion_blobs=[3, 3])
        lat = g.latents("train", [0])
        no_lesion = {**lat, "lesion": torch.zeros_like(lat["lesion"])}
        t1, fl = g.render(lat)
        t1_0, fl_0 = g.render(no_lesion)
        self.assertTrue(torch.equal(t1, t1_0))
        self.assertFalse(torch.equal(fl, fl_0))
        e1, f1 = gen("exact", lesion_blobs=[3, 3]).render(lat)
        e0, _ = gen("exact", lesion_blobs=[3, 3]).render(no_lesion)
        self.assertFalse(torch.equal(e1, e0))

    def test_dependent_and_confounded(self):
        lat = gen("dependent").latents("train", range(400))
        corr = np.corrcoef(lat["g"][:, 0].numpy(), lat["r"][:, :, :].mean(1).numpy().T)[0, 1:]
        self.assertGreater(np.abs(corr).max(), 0.2)
        exact = gen("exact").latents("train", range(400))
        corr0 = np.corrcoef(exact["g"][:, 0].numpy(), exact["r"].mean(1).numpy().T)[0, 1:]
        self.assertLess(np.abs(corr0).max(), 0.2)
        conf = gen("site_confounded").latents("train", range(400))
        site_means = [conf["g"][conf["site"] == s].mean(0) for s in range(6) if (conf["site"] == s).sum() > 5]
        self.assertGreater(float(torch.stack(site_means).std(0).max()), 0.3)

    def test_counterfactual_site_changes_image(self):
        g = gen()
        lat = g.latents("train", [0])
        a = g.render(lat)[0]
        b = g.render(lat, site=(lat["site"] + 1) % 6)[0]
        self.assertGreater(float((a - b).abs().mean()), 0)

    def test_images_normalised(self):
        t1, fl = gen().render(gen().latents("val", range(16)))
        for x in (t1, fl):
            self.assertLess(abs(float(x.mean())), 0.3)
            self.assertGreater(float(x.std()), 0.5)


if __name__ == "__main__":
    unittest.main()
