# Lesion content/style routing

Use this after the lesion on/off reconstruction test shows a localized response
but content probes cannot read lesion location. It tests **decoder reliance** on
content and style, without fitting another readout or changing the representation.

```bash
python -m eval.lesion_routing \
  --run-dir results/synthetic/synthetic-clean-content-causal-baseline-12-4 \
  --num-samples 64 --batch-size 2 --causal iid

python -m eval.lesion_routing \
  --run-dir results/synthetic/synthetic-clean-content-causal-ident-vent-12-4 \
  --num-samples 64 --batch-size 2 --causal iid
```

The script reuses `eval.lesion_reconstruction.render_pair`: same anatomy, style,
acquisition noise, sample seeds and original lesion-on normalization affine.
Only the rendered lesion is removed. The settings supply renderer resolution,
radius and distribution. `--causal iid` deliberately removes SCM correlation;
`--causal match` uses the saved SCM. Donors are always within one subject and one
modality. This is currently restricted to one VQ level, fixed channel selection,
an injected style pathway, and sphere-mode lesions.

## Four decodes

**A = lesion absent; B = lesion present.** The first letter is the content donor,
the second is the style donor. `decode_swaps` from the ventricular diagnostic
supplies all four conditions from the actual forward tensors:

| Condition | Content | Style |
|---|---|---|
| AA | Off | Off |
| BA | On | Off |
| AB | Off | On |
| BB | On | On |

This includes actual quantized content embeddings and the actual decoder-bound
style tensors (quantized if the model quantizes style). It does not reconstruct
content from integer IDs or assume encoder channel indices identify embedding
channels. Separate modality codebooks are supported.

Both unswapped endpoints must reproduce the model's ordinary forward output.
Replay preserves batch shape and disables TF32; replay error is also calibrated
to the local lesion input signal. Substantial mismatch raises an error. Signals
below the replay-error resolution stay in the CSV but are excluded from routing
summaries. Empty/invisible lesion interventions likewise stay in coverage counts.
These checks do not bound every possible numerical error in hybrid decodes.

## Reading the output

All gains project a reconstruction change onto `input_ON − input_OFF` in the
rendered lesion support dilated by one voxel, intersected with foreground. The
dilation accounts for the renderer's 3³ blur. Gains handle bright, dark and mixed
contrast; 1 means an identity response along this direction and 0 means no
projected response. They are **not fractions of information**.

1. Inspect `joint_gain`, `joint_cosine`, and `joint_relative_error` first. Joint
   response is BB − AA. Weak joint fidelity makes routing interpretation weak.
2. `content_at_style_a_gain` measures BA − AA; `content_at_style_b_gain` measures
   BB − AB. These test content in both style contexts.
3. `style_at_content_a_gain` measures AB − AA; `style_at_content_b_gain` measures
   BB − BA. These test style in both content contexts.
4. `content_mean_gain` and `style_mean_gain` average their two donor contexts.
   They sum to joint gain **per subject**, and their sample means add over the
   same subjects. Their medians need not add.
5. `interaction_rms_ratio` measures BB − BA − AB + AA relative to input response
   norm. Large interaction or donor-context differences mean neither pathway has
   a context-independent effect. For example, if the lesion appears only when
   BOTH donors are on, average gains split the effect even though each alone
   produces none. Do not call that independent redundant encoding.

Console output includes means with conditional 95% subject-bootstrap intervals,
medians and coverage. The bootstrap does not capture checkpoint/training-seed
variation. No automatic threshold assigns an exclusive route.

- Content effects near the joint effect, style effects small: decoder dependence
  is predominantly through content for these interventions. Failed content probes
  then need scrutiny; this does not prove easy ordinary-image localization.
- Style effects near joint, content effects small: decoder dependence is
  predominantly through style. It does not prove content contains no information.
- Both substantial, or strong interactions: report shared/context-dependent use.

The CSV also contains quantized code-change fractions and pre/post-codebook RMS
changes. These have different scales and widths; do not compare their magnitudes
as information quantities. Full-foreground absolute-response localization errors
are supplementary; tiny nonzero responses can have apparently good locations.

`responses.png` shows input, both content effects, both style effects and joint
response for **both T1 and FLAIR**. All six images in a row share a signed color
scale. The cross marks the rendered lesion centroid for display only. Use
`--examples 0` to omit plots.

Outputs are `summary.json`, `samples.csv`, and optional `responses.png` in a new
timestamped directory inside the run. `--out-dir` selects a new location; existing
directories are rejected. `--checkpoint` defaults to `vqvae_model.pt`. Batch size
counts subjects: the forward contains four volumes per subject, so two subjects
use eight volumes. Compare only checkpoints with compatible generator settings
and training ages.

Lesion-off images and hybrid latents may be outside the training distribution.
The model stays in evaluation mode, parameters/buffers are restored on exit, and
no checkpoint or optimizer is written. Checkpoint loading is strict. Anatomical
labels define diagnostic interventions/scoring, not a supervised training loss.

```bash
python -m unittest discover -s tests -p 'test_lesion_routing.py' -v
```
