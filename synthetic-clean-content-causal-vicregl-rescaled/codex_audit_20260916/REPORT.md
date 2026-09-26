# VICRegL routing and loss audit

Checkpoint: `synthetic-clean-content-causal-vicregl-rescaled/vqvae_model.pt`, step **34,001**. CPU, 64 IID subjects, existing main audit code at `45cf3c7`. Current working branch was not changed. No model training or checkpoint update.

## Routing

All numbers below are means over replay-resolved subjects; the original ventricle console prints medians. Gains are projections onto the input intervention, not fractions of information or explained variance.

| Intervention | View | Valid | Joint gain | Content gain | Style gain | Joint cosine |
|---|---|---:|---:|---:|---:|---:|
| ventricle | t1 | 62/64 | 0.8236 | 0.000513 | 0.8231 | 0.9093 |
| ventricle | flair | 62/64 | 0.4973 | 0.000067 | 0.4972 | 0.7953 |
| lesion | t1 | 64/64 | 0.4835 | 0.000124 | 0.4834 | 0.7417 |
| lesion | flair | 64/64 | 0.8500 | -0.000073 | 0.8501 | 0.9658 |

Endpoint replay RMS was exactly zero for every subject in both audits. Ventricles had 62/64 measurable interventions per view; lesions had 64/64. The reconstructed intervention response is carried almost entirely by style. Joint fidelity is incomplete, especially T1 lesions and FLAIR ventricles. These tests establish decoder reliance under these interventions, not absence of all information from content, or the historical cause of the routing. IID factors and hybrid codes can differ from the training distribution.

## Logged loss balance

Means of 40 logged steps, 32,050–34,000. Losses below include all actual training coefficients. Local/global diagnostics are on projected features.

| Contribution | Weighted mean |
|---|---:|
| local_sim | 0.716817 |
| local_var | 1.452078 |
| local_cov | 2.111075 |
| global_sim | 0.222598 |
| global_var | 0.110175 |
| global_cov | 0.569209 |
| local_total | 4.279969 |
| global_total | 0.901982 |
| reconstruction | 0.778184 |
| vq | 0.005121 |

Local loss is 82.6% of the logged contrastive objective. It is active; GAP is not dominating the scalar loss. Scalar loss values do not measure encoder gradient magnitudes.

The correct formula is `scale_contrastive_loss × arm_weight × (25 × sim + 25 × var + cov)`. The local/global totals reproduce `Loss/Contrastive` at every logged step through 34,001 (maximum absolute discrepancy 9.54e-07). All TFRecord checksums were verified; all summaries were simple scalar values. CSV training summaries average a logging window whereas TensorBoard values are logged-step values, so they need not match row by row.

Late local projected mean standard deviation is 0.978; global is 1.013. Local variance hinge remains 0.0581: average standard deviation near 1 does not mean every position/channel meets its floor. About 91.8% of already retained positions have enough valid subjects. This eligibility statistic says nothing about coverage of the ventricular boundary or lesions. No NaN-skipped steps were logged.

`Weighted/*` in the existing trainer explicitly decomposes Barlow Twins only. Its residual omits VICRegL and is not evidence of a missing optimization term. Use the VICRegL-specific tags and the corrected breakdown supplied here.

## Next action

Do not increase local weight simply to address this routing: it is already the main contrastive contribution. The objective can be satisfied by other shared anatomical variation; it contains no requirement for ventricular size or lesions specifically. Full spatial style (`style_spatial_size: 0`) provides a reconstruction route for them. This is a plausible mechanism supported by the intervention results, not proof of training causality.

The next controlled unsupervised ablation is the same VICRegL configuration with only `style_spatial_size: 1` changed, using a new run/tag. Pooling style constrains its spatial capacity but does not guarantee that global anatomical factors cannot enter it. Keep the projection heads and loss weights initially. Evaluate at an early fixed checkpoint with these same routing tests; count it as improvement only if content response increases while joint response fidelity remains acceptable. A rise in content share caused by degraded reconstruction is not success.

A local directory named `synthetic-clean-content-causal-sp-s-1` is present, but its settings use Barlow Twins with outer scale 100. It is not the matched VICRegL style-size ablation.

For strictly matched training-distribution interpretation, also repeat these audits with `--causal match` before making a broader causal claim. The present IID tests deliberately remove factor correlations.

## Reproduction

Run from code at the recorded commit (the current working branch lacks these audit modules). Substitute the run directory on the training machine:

```bash
RUN=results/synthetic/synthetic-clean-content-causal-vicregl-rescaled
python -m eval.ventricle_routing --run-dir "$RUN" --checkpoint vqvae_model.pt --num-samples 64 --batch-size 2 --eps 0.25 --causal iid
python -m eval.lesion_routing --run-dir "$RUN" --checkpoint vqvae_model.pt --num-samples 64 --batch-size 2 --causal iid
```

Local artifacts: `ventricle/summary.json`, `ventricle/samples.csv`, `lesion/summary.json`, `lesion/samples.csv`, `routing_means.json`, `loss_balance/scalars.csv`, `loss_balance/weighted_terms.csv`, `loss_balance/loss_balance.png`, and `provenance.json`.
