# Normative bundle integration smoke test — 2026-10-14

This is a small pipeline check, not a reproduction of the research-scale results in the
design note. Source run: `results/synthetic/synthetic-clean-content-causal-proj` (64³ inputs).
Normative fit: 96 training subjects, 8 modes per view, 10 null permutations, spatial ICA,
6 explicitly requested shared dimensions, 3 residual peak channels, per-view output.
The automatic width at this small fit size was 4; forcing 6 provides enough channels for
the 9-factor graph integration check. Evaluation: 64 test subjects. Export took 13 seconds.

- `normative.npz`: normative bundle and accompanying `.meta.json`.
- `vq.npz`: content/GAP bundle from the source run's existing VQ checkpoint.
- `compare.json` / `compare.txt`: shared probe protocol, seed 0, 3 folds, 2 permutations.
- `graph.json`: direct-channel PC on both bundles, alpha 0.05, 20 naming-null draws,
  3 mixing draws per representation, one truth ceiling. Normative mixing leaves columns
  6–8 (lesion coordinates) fixed; VQ mixing uses its 12 content columns.

Both bundles have exactly equal factor and generator digests:

```
factor:    f60ba356057dd49a3916fe40d1ca6b1b59aeabfbacdec4bcd47b8808dc2c30d3
generator: 66cbd74245c270418f43ad62b9a1bc6f0d8466f5474ca724c06450557fa29b09
```

Validation used `/opt/miniconda3/envs/adni-analysis/bin/python`, with
`VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` and a writable
`MPLCONFIGDIR`. Both module self-tests and all 19 new regression tests pass.
The 135-test bundle/causal/identifiability run has 134 passes and one existing failure:
`test_dinov3_identifiability.IndepTestTests.test_capping_the_conditioning_set_only_adds_skeleton_edges`.
It also fails in isolation through unchanged modules. Installed `causal-learn 0.1.4.0`
accepts `max_k` through `**kwargs` but its skeleton discovery has no conditioning-depth
argument or implementation. This smoke run uses the default uncapped search.

Reproduce the export:

```bash
python -m eval.protocol.export_normative_bundle \
  --run-dir results/synthetic/synthetic-clean-content-causal-proj \
  --fit-samples 96 --num-samples 64 --components 8 --shared-dims 6 \
  --null-permutations 10 --batch-size 8 \
  --out out/normative_bundle_smoke_20261014/normative.npz
```

See `eval/protocol/NORMATIVE_BUNDLE.md` for the regular-size workflow and interpretation.
