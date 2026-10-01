# Encoder architecture ablations on slurm_bio

These jobs run the same three encoder interventions, optional controls and
training recipe as the [Run:ai ablations](ENCODER_ABLATIONS_RUNAI.md). They use
SLURM batch jobs and the cluster's existing Python environment.

## Submit from the repository root

Sync the updated repository (including model changes, tests and generated
scripts) to the cluster first. From its root, submit:

```bash
sbatch experiments/generated/encoder_conv_mlp_s42.slurm_bio.sh
sbatch experiments/generated/encoder_resnet_groupnorm_s42.slurm_bio.sh
sbatch experiments/generated/encoder_resnet_stride8_s42.slurm_bio.sh
```

Each command submits one independent job. The interventions are:

- Conv: replace the linear readout with GAP → MLP(64→100→12).
- ResNet: replace all BatchNorm layers with 32-group GroupNorm.
- ResNet: remove layer3/layer4 downsampling, producing an 8³ final map from
  64³ inputs. The original stem stays intact; no dilation is added.

Optional controls under the same cluster environment:

```bash
sbatch experiments/generated/encoder_conv_s42.slurm_bio.sh
sbatch experiments/generated/encoder_resnet18_s42.slurm_bio.sh
```

Compare Conv MLP against native Conv, and each modified ResNet against native
ResNet. Keep data, code and environment matched within each comparison.

To preview a command locally without submitting, loading modules or invoking
Python:

```bash
bash experiments/generated/encoder_conv_mlp_s42.slurm_bio.sh --dry-run
```

Normal execution requires a SLURM allocation. Use `sbatch`, rather than `bash`,
to submit. The job uses `SLURM_SUBMIT_DIR` as its repository directory; that is
the directory where `sbatch` was invoked. See the
[SLURM reference](https://slurm.schedmd.com/sbatch.html#OPT_SLURM_SUBMIT_DIR).
For submission from elsewhere, set `ENCODER_REPO` to the absolute repository
path before submitting, or generate with `--repo-path`.

## Resources and environment

The generator reads `experiments/cluster/slurm_bio.yaml`, including its `_slurm`
resource block. It also accepts `_slurm_bio` when supplied in a custom YAML.
The supplied scripts request:

| Setting | Value |
|---|---|
| Partition | `biomed_a100_gpu` |
| GPU | One, with `a100_80g` constraint |
| Nodes / tasks | One / one |
| CPUs | 8 |
| RAM | 64G |
| Wall time | 48 hours |
| Module | `anaconda3/2022.10-gcc-13.2.0` |
| Python | `$HOME/.conda/envs/multiview-env/bin/python` |

Jobs reuse the prepared environment. They never remove Conda environments or
install packages, so concurrent jobs cannot alter each other's dependencies.
Prepare the project's training dependencies before submission. To use another
existing Python environment, export its full interpreter path, for example:

```bash
export ENCODER_PYTHON="$HOME/.conda/envs/monai_env/bin/python"
sbatch experiments/generated/encoder_conv_mlp_s42.slurm_bio.sh
```

The job checks CUDA and runs the determinism/max-pooling regression tests before
training. Missing Python, unavailable CUDA or failing checks stop the job. It
uses deterministic algorithms with warning fallback, as in the Run:ai jobs.
All numerical thread counts stay at one to preserve the paired data-rendering
protocol, even though eight CPUs are allocated.

Stdout and stderr go to `/scratch/users/%u/%x-%j.out` and `.err`: `%u` is the
username, `%x` the job name and `%j` the job ID. That user scratch directory
must already exist when SLURM opens the logs.

## Training and outputs

Defaults remain seed 42, 64³ paired T1/FLAIR volumes, 2,000 training subjects,
400 validation subjects, batch 32 per view, InfoNCE with temperature 0.1,
AdamW at 1e-4, 10,000 updates and validation every 2,000 updates. Model seed is
42, loader seed 10042 and data seed 42. Input and batch-order hashes are saved.

Results use a separate root from the Run:ai runs:

```text
results/encoder_ablations_slurm_bio/runs/
  conv_s42/                 # optional control
  resnet18_s42/             # optional control
  conv_mlp_s42/
  resnet_groupnorm_s42/
  resnet_stride8_s42/
```

Existing runs are refused. These scripts start fresh training and do not resume
optimizer state. They perform periodic validation but not the full post-training
audit. Use the [same evaluation commands](ENCODER_ABLATIONS_RUNAI.md#evaluate-completed-checkpoints)
with, for example, `RUN=results/encoder_ablations_slurm_bio/runs/conv_mlp_s42`,
inside a GPU allocation after training completes.

## Regenerate

Python and PyYAML are sufficient for generation. Generation does not submit jobs:

```bash
python scripts/generate_encoder_ablation_slurm.py --include-controls
```

Generate all three configured seeds (15 scripts including controls):

```bash
python scripts/generate_encoder_ablation_slurm.py \
  --seeds 42 142 242 --include-controls
```

Use `--config` for a copy of the matched recipe, `--cluster-config` for another
resource YAML, `--repo-path` for the cluster repository, and `--results-dir` for
another output root. For a fresh attempt:

```bash
python scripts/generate_encoder_ablation_slurm.py --include-controls \
  --results-dir results/encoder_ablations_slurm_bio_v2 \
  --job-prefix encoder-ablation-bio-v2
```

Regenerate after editing resource or training settings. Other generated seeds
remain in place; choose submission files explicitly.

## Checks

```bash
python -m unittest tests.test_encoder_ablation_slurm tests.test_encoder_ablation_runai
```

Checks validate resource directives, training-option parity, paths with spaces,
seed generation, and failure propagation using fake local module/Python commands.
They do not contact SLURM, execute CUDA training or establish cluster memory use.
