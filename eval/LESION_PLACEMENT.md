# Full-sphere white-matter lesion placement

Enable the corrected generator for a **new** experiment:

```bash
python -m training.main_conv_synthetic \
  --model-id dummy_infonce_wm_interior \
  --res 64 --downscale-factor 4 \
  --latent-dim 12 --content-channels 9 --n-content 9 --n-style 3 \
  --tau 0.1 --batch-size 64 --hidden-channels 64 --no-cache \
  --synthetic-normalize fixed_reference --synthetic-clean-content \
  --synthetic-lesion-placement wm_interior \
  --num-train-samples 2000 --num-val-samples 400 \
  --train-steps 50000 --eval-every 2000
```

For VQ-VAE training, add `synthetic_lesion_placement: wm_interior` to the YAML
(or use the same CLI option). The mode is recorded in settings and forwarded by
the dense trainer, `eval.score_checkpoint`, the main VQ-VAE trainer, the common
synthetic evaluator and signature-driven intervention/probe dataset factories.
No supervision is added to model training.

## Geometry and latent control

The previous `legacy` sphere placement constrains support using a geometric WM
envelope before CSF and fissure labels overwrite it. Therefore it can place
lesions inside CSF. `wm_interior` instead:

1. Finds the **final** white-matter label (`tissue_map == 2`).
2. Uses a distance transform to select voxel centres with clearance greater than
   the requested radius from every non-WM voxel, including padded volume edges.
3. Maps the three squashed lesion-position latents to conditional x/y/z quantiles
   of those admissible centres. No random seed or rejection sampling selects the
   centre. Each axis latent controls its corresponding coordinate; the y/z choices
   are conditional on earlier coordinates so they remain within admissible WM.
4. Draws the complete sphere at that centre with the original radius. It never
   intersects/truncates the resulting sphere with a tissue mask or shrinks it.

Containment applies to the voxelized anatomy: every lesion voxel has underlying
WM tissue. It is not a claim about continuous subvoxel boundaries. Image blur can
subsequently spread lesion signal across tissue boundaries, as before.

Centres lie on voxel centres, making the discretized sphere volume constant at a
given radius and resolution. At radius 0.1 and resolution 64 it contains 123
voxels. This differs slightly from subvoxel-centred legacy spheres (typically
around 131), even though the nominal radius is unchanged.

The latent-to-position map is now an ordered, discretized mapping onto valid WM,
not the old linear scaling of squashed xyz. Its admissible region depends on
anatomy. Consequently physical-centroid probes are useful alongside latent-xyz
probes, and changing ventricular geometry can relocate a lesion even when its
position latents stay fixed. Existing ventricle-only diagnostics that require an
identical lesion load at both endpoints will reject such pairs; their existing
guard is intentional. Do not interpret those coupled changes as ventricle-only
responses.

## Compatibility and impossible placements

The default remains `legacy` so saved runs with no placement setting reproduce
their original images. Both old and new modes are explicit CLI choices. Use a
new run directory when enabling `wm_interior`; changing the generator underneath
an old checkpoint is not a matched evaluation.

If no complete voxel-centred sphere fits a subject's anatomy, the renderer raises
`LesionPlacementError` (a `ValueError`) reporting radius, resolution and maximum
clearance. It never truncates, shrinks or removes a lesion. The dataset responds by
**redrawing that subject**: it tries further deterministic candidate draws (all
latents, not just the lesion) and keeps the first whose anatomy fits, warning once
per redrawn subject. Subjects that fit on their original draw are byte-identical to
before. `sample_seed_for(idx)` returns the accepted candidate's seed, so evaluators
that re-render a subject reproduce it exactly. If none of
`MAX_LESION_RESAMPLES + 1` candidates fits, the error says the radius is too large
for the anatomy distribution.

Redrawing conditions the anatomy distribution on "a complete lesion fits". For
`experiments/synthetic_causal.yaml` at radius 0.1 (res 64) that replaces 2 of 3,900
subjects (train 559, val 1429: small brains with enlarged ventricles, 3.00 voxels of
clearance against a 3.15-voxel radius). At res 64 the voxelized lesion jumps from 123
to 93 voxels at radius 0.0952, so no radius in between keeps the size. Out-of-range
position values stay a plain `ValueError` and are never redrawn: that is a
configuration error. Latents passed explicitly to `render_pseudo_mri` (for example
interventions) are not redrawn either; they raise as before. Field-lesion mode is
unchanged; combining it with `wm_interior` is rejected because this option places spheres.

For the supplied dense-run distribution (seed 42, clean IID content, resolution
64, radius 0.1, default ventricles), all **2,000 training + 400 validation + 400
test** geometries passed, with zero non-WM overlap and 123 lesion voxels each.
This scan uses the same first nine latent draws per sample as the dataset; the
clean setting removes deformation/fissure nuisance. It does not train a model.
An additional 128-subject scan with `identifiable_ventricle=True` found two
anatomies where no sphere of radius 0.1 fits; such subjects are now redrawn (see
above). How many a setting redraws depends on anatomy, radius and resolution, so
check the warnings when changing any of them.

Tests cover complete sphere geometry, both ventricular variants, deterministic
rendering, each position coordinate, disconnected WM, boundary cases, redrawing of
subjects with no room (reproducible via `sample_seed_for`), bounded failure when no
candidate fits, unchanged legacy behavior and train/eval propagation:

```bash
python -m unittest discover -s tests -p 'test_lesion_placement.py' -v
```
