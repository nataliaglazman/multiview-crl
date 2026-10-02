# Three SLURM encoder ablations: seed 42

All three supplied stdout logs finish 10,000 optimizer steps, print the final
validation table, and report saved checkpoints. Conv + MLP has the strongest
recovery of broad anatomy in these logs. None demonstrates useful learned
recovery of lesion coordinates or sulcal widening in the printed global content
representation. Both ResNet variants reduce their training loss while their
average validation ridge recovery ends below its untrained value.

## Evidence and scope

| Run | Job | Final evaluation | Completion line |
|---|---|---|---|
| Conv + MLP | 37697225 | [stdout line 301](logs/encoder-ablation-bio-conv-mlp-s42-37697225.out#L301) | line 328 |
| ResNet + GroupNorm | 37697226 | [stdout line 301](logs/encoder-ablation-bio-resnet-groupnorm-s42-37697226.out#L301) | line 328 |
| ResNet stride 8 | 37697227 | [stdout line 301](logs/encoder-ablation-bio-resnet-stride8-s42-37697227.out#L301) | line 328 |

Each log contains 100 training summaries and six evaluations (steps 0, 2000,
4000, 6000, 8000, 10000). These show trainer completion, not an independently
queried SLURM exit status. The separate stderr files, checkpoint files, and
training-progress JSON files were not among the supplied artifacts.

The printed settings agree on data seed 42, model seed 42, loader seed 10042,
64³ inputs, 2,000 training and 400 validation subjects, batch 32 per view, InfoNCE
temperature 0.1, AdamW learning rate 1e-4, 10,000 updates, and separate view
backbones. Every job reports PyTorch 2.3.1+cu121 and CUDA 12.1, on different nodes.
The only differing printed option fields are model_id, encoder_architecture,
conv_readout, resnet_norm and resnet_output_stride. Some architecture-specific
flags are inactive in the other architecture. Source hashes, actual input/order
hashes, and GPU models are not printed, so full pairing cannot be verified here.

These are validation probes, not the independent post-training test audit.
The displayed content→content factor table reads the first view's nine global
content coordinates. It is not an average over both views. The current evaluator
also computes second-view scores, but the trainer does not print those tables;
the saved dci_step10000.json contains the additional scalar results. The configured
pooling is GAP; the printed eval_patch_grid does not mean patches were scored.

The ridge implementation uses five-fold cross-validation with three probe split
seeds. Those splits are not three independently trained models. The table's ±
column belongs to block-MCC, not ridge R², and does not quantify training-seed
uncertainty. Values below retain stdout's three-decimal precision. The 'floor'
is each model's own untrained initialization, not a shuffled-target null.

## Final recovery

| Factor: validation ridge R² | Conv + MLP | ResNet + GroupNorm | ResNet stride 8 |
|---|---:|---:|---:|
| Brain size | 0.918 | -0.012 | 0.060 |
| Ventricle size | 0.818 | -0.022 | -0.027 |
| Lesion x | -0.017 | -0.015 | -0.021 |
| Lesion y | -0.015 | -0.017 | -0.018 |
| Lesion z | -0.015 | -0.010 | -0.012 |
| Cortical thickness | 0.824 | 0.086 | 0.222 |
| Temporal atrophy | 0.798 | 0.479 | 0.528 |
| Left/right asymmetry | 0.886 | 0.111 | 0.135 |
| Sulcal widening | -0.015 | 0.032 | -0.016 |
| **Mean across nine factors** | **0.465** | **0.070** | **0.094** |

![Final recovery](final_factor_recovery.png)

Near-zero or negative held-out R² is not useful recovery by this ridge probe;
it does not prove that every representation of the model lacks the information.
All nine final lesion-coordinate scores are negative. No model shows sustained
lesion learning over the six printed evaluations. Conv + MLP has a small lesion-x
score of 0.023 at step 2000, which disappears later.

The small GroupNorm sulcal score does not establish learning: its own initialization
already scored 0.029. The logged increase is only +0.004 (subtraction of the rounded
displayed scores gives 0.003). The other two models finish at negative sulcal R².

## Training dynamics and initial baselines

| Metric | Conv + MLP | ResNet + GroupNorm | ResNet stride 8 |
|---|---:|---:|---:|
| Initial mean ridge R² | 0.115 | 0.116 | 0.168 |
| Final mean ridge R² | 0.465 | 0.070 | 0.094 |
| Change, using rounded means | +0.350 | -0.046 | -0.074 |
| Final training loss | 0.4550 | 0.2490 | 0.1823 |
| Final training effective rank / 9 | 6.19 | 7.00 | 7.21 |
| Final block-MCC | 0.583 | 0.267 | 0.302 |
| Final channel-MCC | 0.384 | 0.153 | 0.177 |
| Final GBT informativeness | 0.344 | -0.025 | 0.066 |
| Final DCI disentanglement | 0.140 | 0.028 | 0.028 |
| Final DCI completeness | 0.161 | 0.032 | 0.043 |
| Final view-classification accuracy | 0.537 | 0.475 | 0.474 |

Training loss/rank entries are the final 100-step averages, not evaluation-set
measurements. GBT informativeness uses a different probe and split protocol from
ridge and should not be treated as a directly matched alternative.

![Training and recovery](training_and_recovery.png)

Conv + MLP mean recovery rises 0.115 → 0.291 → 0.393 → 0.452 → 0.467 → 0.465.
The small difference between 8,000 and 10,000 does not establish significant
deterioration. Its five broad anatomy factors finish between 0.798 and 0.918.
The model still has weak axis alignment/disentanglement and fails the focal targets.

The ResNet decline is already visible at the first trained checkpoint:

| Factor | GroupNorm: initial → 2k → 10k | Stride 8: initial → 2k → 10k |
|---|---|---|
| Brain size | 0.407 → 0.000 → -0.012 | 0.751 → 0.082 → 0.060 |
| Left/right asymmetry | 0.368 → 0.283 → 0.111 | 0.664 → 0.243 → 0.135 |
| Temporal atrophy | 0.059 → 0.453 → 0.479 | 0.036 → 0.589 → 0.528 |

Thus training improves some factors while making others substantially less
linearly recoverable. This is more specific than saying the networks failed to
learn. Temporal atrophy reaches 0.633/0.647 at 4,000 steps in GroupNorm/stride 8,
then ends lower. These curves alone do not identify overfitting as the cause.

The ResNets have lower training loss and higher training effective rank, yet much
lower validation factor recovery than Conv + MLP. Neither quantity is an adequate
checkpoint-selection proxy for the target factors. High rank rules out complete
constant/one-dimensional collapse on those training batches; it does not certify
information content, generalization, or healthy eval-mode feature statistics.

All three reduce view-classification accuracy towards 0.5 and their final
content→style ridge scores are negative for gain, bias and noise. That is consistent
with reduced accessibility of view/style information to these probes. It does not
prove full independence, nor guarantee retention of the desired content factors.

## What the architectural interventions establish

- **Conv + MLP:** a nonlinear readout is compatible with good broad-anatomy recovery
  in this run. It does not solve lesion/sulcal recovery. There is no matched native
  Conv control stdout here, so the effect of adding the MLP is not established.
- **ResNet + GroupNorm:** replacing BatchNorm does not rescue this run's measured
  factor recovery. A BatchNorm running-statistics issue cannot explain this
  GroupNorm model's failure; it may still contribute to the stride-8 model's scores.
- **ResNet stride 8:** retaining an 8³ map does not rescue the printed globally
  pooled representation. This does not rule out recoverable spatial information:
  the readout still pools globally, and the stride-2 stem and max-pool remain.

The native Conv and native ResNet control logs are absent. The previously pasted
native ResNet mean R² of 0.161 is numerically above both ResNet variants here, but
code/environment/input pairing has not been verified. The previously pasted Conv
result was also at a different training step. Neither historical comparison is
sufficient to claim a causal effect of normalization, readout or stride. These
are single-training-seed results, with no between-seed error bars.

## Next diagnostic on the existing checkpoints

Before new training, locate where recoverability falls using both views and the
existing stage audit: pooled backbone → hidden readout → final content, with
ridge and RBF probes and train/test retrieval. High train retrieval with poor test
retrieval would support a generalization problem. Good backbone recovery with
poor final-content recovery would implicate the readout/content projection.

Run inside a GPU allocation in the cluster repository. This loop evaluates
existing checkpoints; each audit chooses its own timestamped output directory.

```bash
set -euo pipefail
BASE=results/encoder_ablations_slurm_bio/runs
for MODEL in conv_mlp_s42 resnet_groupnorm_s42 resnet_stride8_s42; do
  python -m eval.encoder.encoder_generalization_audit \
    --run-dir "$BASE/$MODEL" --checkpoint model.pt \
    --batch-size 4 --num-samples 400 --probe-samples 400 \
    --retrieval-draws 8 --seed 1729 --skip-bn-recalibration
done
```

Keep original checkpoint results primary. Separately, stride 8 can be audited
without `--skip-bn-recalibration` to compare a disposable copy with refreshed
BatchNorm statistics; that does not retrain weights or overwrite the checkpoint.

For lesions, use the existing spatial/centroid diagnostic on common grids 1 and 2
first, with matched untrained and shuffled-target controls. Larger grids 4 and 8
are additional diagnostics available for stride 8, not a common comparison with
stride-32 GroupNorm. Better spatial than GAP recovery would support a pooling
bottleneck. Failure at both stages still calls for checking rendered target
visibility and early encoder stages before attributing the problem to the loss.

## Reproducible extraction

`analyze_logs.py` parses the original logs without changing them, checks expected
step counts, checks per-factor means against printed summary values, checks
initial floors and rounded deltas, and verifies that only the five expected
configuration fields differ. It writes `parsed_logs.json` with source SHA-256
hashes, `factor_recovery.csv` with all 216 factor rows, and the two figures above.

All conclusions here are based on the supplied stdout and inspection of the local
metric implementation. No new training or checkpoint evaluation was launched.
