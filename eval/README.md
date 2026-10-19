# eval/

Run any CLI from the repo root as `python -m eval.<subpackage>.<module>`.
Each topic's write-ups (`*.md`) live next to its code.

| Subpackage | What's in it |
|---|---|
| `metrics/` | Metric primitives other code imports: `dci`, `identifiability_metrics`, probe helpers (`evaluation`), `cross_reconstruction`, `marginal_independence`, `parent_adjusted`, `bundle_identity`, `init_baseline`. |
| `protocol/` | Scoring entrypoints. `run_dci_compare` is the source of truth for the metric rules; `identifiability_report`, `score_checkpoint`, `run_dci_synthetic`, `compare_bundles`, `export_vq_bundle`, [normative content bundles](protocol/NORMATIVE_BUNDLE.md), `interventional_identifiability`, VAE sweep scoring; `spatial_bottleneck_eval` (per-factor recovery from a [spatial-bottleneck](../training/SPATIAL_BOTTLENECK.md) code vs the full map, GAP, patch, PCA at the same size and the untrained readout). |
| `synthetic/` | The synthetic generator (`synthetic_dataset`, `legacy_renderer`), its checkpoint-free audits (`generator_defects`, `factor_visibility`), previews and NIfTI export. |
| `causal/` | PC causal discovery on representations (`run_causal_recovery`, `latent_causal_discovery`) and SCM diagnostics. |
| `dino/` | DINOv3 / 3DINO embedding and scoring, plus the DINO-vs-VQ comparison notes. |
| `gradients/` | Per-term gradient audits of the real training objective (`gradient_attribution`, `loss_gradient_audit`), BT term balance, loss breakdowns. |
| `lesion/` | Lesion probes, routing, pooling, placement, alignment and checkpoint lesion analysis; `lesion_miss_location` (do the lesions a `lesion_reconstruction` audit loses in the reconstruction share a location: white-matter depth, distance to grey matter, CSF and the midline; with `--predictions`, whether a spatial-bottleneck evaluation decodes those lesions worse than the rest); `lesion_feature_maps` (figures of where a lesion sits in a VQ-VAE's frozen feature maps: lesion-on minus lesion-off per channel, footprint locality, style share and sign coherence). |
| `ventricle/` | Ventricle routing, pooling, alignment, decoder/quantizer audits, the central-ventricle probe. |
| `encoder/` | Encoder-only ablation audits (generalization, spatial targets, target-control protocol); [GN/LN normalization and pooling probe](encoder/ENCODER_NORMALIZATION_AUDIT.md), [early-layer lesion movement test](encoder/ENCODER_LESION_NORM_AUDIT.md), [anatomy within vs across `--stratify-batches` clusters](encoder/within_cluster_probe.py), [batch code standardization and anatomy reliance](encoder/CODE_STANDARDIZATION_PROBE.md). |
| `maps/` | Voxelwise/spatial maps: `recovery_maps`, `identifiability_maps`, phase-0 extraction, SPM, receptive-field and radial-profile plots. |
| `diagnostics/` | One-question checkpoint/representation probes: style/content path audits, pooling and patch-signal probes, norm and BT calibration, background/view leak, checkpoint forensics, entropy/uniformity, patch-MCC decay; `attention_template_overlap` (do a `--global-pool attention` run's heads sit on the synthetic temporal-atrophy site, against same-shape templates elsewhere); `attention_pool_gradients` (where the real training loss pushes each attention head's logits at a checkpoint, how consistently across the trainer's batches, and whether it rewards that site against the same templates; also each query gradient split by view, and whether finite logit offsets toward the site lower the step loss beyond first order). |
| `adni/` | Real-ADNI probes: disease classifier, anatomy and channel probes; `label_probes_vs_init` (an encoder run's diagnosis/sex/race probes at its best checkpoint vs step 0, from the run dir or stdout); [encoder modality-gap diagnostic](adni/ENCODER_MODALITY_GAP.md) and [its candidate-cause tests](adni/ENCODER_MODALITY_GAP_CAUSES.md). |
| `plots/` | Figure scripts that read saved JSON and never re-score. |
| `notebooks/` | Analysis notebooks (they add `../..` to `sys.path`). |

## Temporal atrophy width

`--synthetic-temporal-sigma 0.12` (or YAML `synthetic_temporal_sigma: 0.12`)
narrows the bilateral temporal-atrophy Gaussian. The default `0.18` preserves
existing images; `0.10` is narrower still. This changes spatial extent while
leaving the strength coefficient and centres fixed. With local-deformation
centring enabled, a small compensating deformation still extends elsewhere.
At small widths, fewer tissue boundaries are reached, so the visible effect
and total image signal also decrease.

Render matched T1/FLAIR slices and difference maps using the actual generator:

```bash
python -m eval.synthetic.preview_temporal_atrophy --out-dir out/temporal_width
```

The preview holds every other latent and acquisition seed fixed, compares
`z6=0` with `z6=1` at widths `0.18`, `0.12`, and `0.10`, and saves raw volumes,
tissue maps and comparison settings. Add `--center-local-deformations` to
inspect the compensation, or `--sigmas 0.18 0.14 0.12` to choose other widths.

## Removed scripts

Deleted in commit `8e1341c`; recover any with
`git show 8e1341c~1:eval/<name>.py > eval/<subpackage>/<name>.py`.

- Notebook copies: `analyze_synthetic_recovery copy*.ipynb` (6), `identifiability_report copy.ipynb`.
- `_make_recovery_nb.py`: stale generator, regenerating deleted Sections 7g–17 of the recovery notebook.
- `receptive_field_test.py`: backprop RF through GroupNorm always spans the volume, so it measured the norm, not the conv RF.
- `_check_lesion_field.py`, `find_jump_cause.py`: one-off debugging.
- MCC-decay mechanisms that were measured and refuted: `smuggling_test`, `vq_resolution`, `dc_channel_test`, `groupnorm_carrier_test`, `direction_diffusion` (+ its helpers `alignment_scale_test`, `view_asymmetry_probe`), `labeled_variance_share`, `factor_correlation_fidelity`, `norm_gamma_trace`, `patch_variance_split`.
