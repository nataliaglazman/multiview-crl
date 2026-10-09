# eval/

Run any CLI from the repo root as `python -m eval.<subpackage>.<module>`.
Each topic's write-ups (`*.md`) live next to its code.

| Subpackage | What's in it |
|---|---|
| `metrics/` | Metric primitives other code imports: `dci`, `identifiability_metrics`, probe helpers (`evaluation`), `cross_reconstruction`, `marginal_independence`, `parent_adjusted`, `bundle_identity`, `init_baseline`. |
| `protocol/` | Scoring entrypoints. `run_dci_compare` is the source of truth for the metric rules; `identifiability_report`, `score_checkpoint`, `run_dci_synthetic`, `compare_bundles`, `export_vq_bundle`, `interventional_identifiability`, VAE sweep scoring. |
| `synthetic/` | The synthetic generator (`synthetic_dataset`, `legacy_renderer`), its checkpoint-free audits (`generator_defects`, `factor_visibility`), previews and NIfTI export. |
| `causal/` | PC causal discovery on representations (`run_causal_recovery`, `latent_causal_discovery`) and SCM diagnostics. |
| `dino/` | DINOv3 / 3DINO embedding and scoring, plus the DINO-vs-VQ comparison notes. |
| `gradients/` | Per-term gradient audits of the real training objective (`gradient_attribution`, `loss_gradient_audit`), BT term balance, loss breakdowns. |
| `lesion/` | Lesion probes, routing, pooling, placement, alignment and checkpoint lesion analysis. |
| `ventricle/` | Ventricle routing, pooling, alignment, decoder/quantizer audits, the central-ventricle probe. |
| `encoder/` | Encoder-only ablation audits (generalization, spatial targets, target-control protocol); [GN/LN normalization and pooling probe](encoder/ENCODER_NORMALIZATION_AUDIT.md), [early-layer lesion movement test](encoder/ENCODER_LESION_NORM_AUDIT.md). |
| `maps/` | Voxelwise/spatial maps: `recovery_maps`, `identifiability_maps`, phase-0 extraction, SPM, receptive-field and radial-profile plots. |
| `diagnostics/` | One-question checkpoint/representation probes: style/content path audits, pooling and patch-signal probes, norm and BT calibration, background/view leak, checkpoint forensics, entropy/uniformity, patch-MCC decay. |
| `adni/` | Real-ADNI probes: disease classifier, anatomy and channel probes; [encoder modality-gap diagnostic](adni/ENCODER_MODALITY_GAP.md). |
| `plots/` | Figure scripts that read saved JSON and never re-score. |
| `notebooks/` | Analysis notebooks (they add `../..` to `sys.path`). |

## Removed scripts

Deleted in commit `8e1341c`; recover any with
`git show 8e1341c~1:eval/<name>.py > eval/<subpackage>/<name>.py`.

- Notebook copies: `analyze_synthetic_recovery copy*.ipynb` (6), `identifiability_report copy.ipynb`.
- `_make_recovery_nb.py`: stale generator, regenerating deleted Sections 7g–17 of the recovery notebook.
- `receptive_field_test.py`: backprop RF through GroupNorm always spans the volume, so it measured the norm, not the conv RF.
- `_check_lesion_field.py`, `find_jump_cause.py`: one-off debugging.
- MCC-decay mechanisms that were measured and refuted: `smuggling_test`, `vq_resolution`, `dc_channel_test`, `groupnorm_carrier_test`, `direction_diffusion` (+ its helpers `alignment_scale_test`, `view_asymmetry_probe`), `labeled_variance_share`, `factor_correlation_fidelity`, `norm_gamma_trace`, `patch_variance_split`.
