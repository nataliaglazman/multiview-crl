# Run encoder experiments on any backend

Every experiment in [`experiments/encoder`](../../encoder/) has four generated scripts:
`<name>.local.sh`, `<name>.slurm.sh`, `<name>.slurm_bio.sh`, and `<name>.runai.sh`.
The model, data recipe, seeds, and training schedule come from the same YAML. Cluster
profiles supply resource requests and, for ADNI, data paths and output locations.

## One command, different backends

From the repository root, in your Python environment:

```bash
python scripts/launch.py experiments/encoder/conv_mlp.yaml --cluster local --dry-run
```

Change `local` to `slurm`, `slurm_bio`, or `runai`; remove `--dry-run` to run or submit.
This interface reads the current YAML and supports `--set key=value` on every backend:

```bash
# Apple Silicon: explicitly select MPS (the default auto selects CUDA or CPU).
python scripts/launch.py experiments/encoder/conv_mlp.yaml --cluster local --set device=mps

# Another paired seed; the tag changes automatically to conv_mlp_s142.
python scripts/launch.py experiments/encoder/conv_mlp.yaml --cluster slurm_bio --set seed=142
```

The launcher and generation need Python 3.10+ and PyYAML. Local training also needs the
encoder training dependencies in that environment. Local launches use that Python,
enable CPU fallback when MPS is selected, and use `caffeinate` on macOS to prevent idle sleep.

## Use the generated scripts

These contain the resolved recipe from their last generation. Run the following from
the repository root (substitute any experiment name for `conv_mlp`):

| Backend | Run or submit |
|---|---|
| Local | `bash experiments/generated/encoder/conv_mlp.local.sh` |
| SLURM, `gpu` partition | `sbatch experiments/generated/encoder/conv_mlp.slurm.sh` |
| SLURM Bio, `biomed_a100_gpu` partition | `sbatch experiments/generated/encoder/conv_mlp.slurm_bio.sh` |
| Run:ai | `bash experiments/generated/encoder/conv_mlp.runai.sh` |

Preview **any** script with `bash <script> --dry-run`. Use `bash`, including for SLURM
previews, so no job is submitted. SLURM and Run:ai previews/submission scripts need no Python
on the submit host; their training environments still need it. Sync the repository to
the cluster before submitting. Cluster settings live in [`experiments/cluster`](../../cluster/).

Local scripts find the checkout from their own location, so they also work when invoked
from another directory. Set `ENCODER_REPO` if you move a script elsewhere, or
`ENCODER_PYTHON=/path/to/python` to select an interpreter. Extra local arguments are trainer
flags (hyphenated names), applied after the generated recipe:

```bash
bash experiments/generated/encoder/conv_mlp.local.sh --device mps --dry-run
bash experiments/generated/encoder/conv_mlp.local.sh --device cuda
```

For a smaller local trial, explicitly change the batch and give it a new run ID:

```bash
bash experiments/generated/encoder/conv_mlp.local.sh \
  --device mps --batch-size 8 --model-id conv_mlp_b8_s42
```

Changing batch size changes the experiment. Keep it matched across compared arms. Generated
scripts have fixed run IDs: when overriding `--seed` locally, also override `--model-id`, or
use the YAML launcher to derive the ID automatically. Existing run directories are refused;
use a new tag/model ID or output directory for a fresh attempt.

## Local ADNI data

The three `adni*.local.sh` scripts require your paths at runtime. For example:

```bash
bash experiments/generated/encoder/adni.local.sh --device mps \
  --dataroot /path/to/data \
  --labels-path /path/to/labels_cleaned_3class_demog.csv \
  --masks-dir /path/to/ADNI_stripped_masks \
  --dry-run
```

Remove `--dry-run` to train. The YAML interface accepts the same paths via
`--set dataroot=... labels_path=... masks_dir=...`. See
[`ENCODER_ADNI.md`](../../../training/ENCODER_ADNI.md) for the data layout and preprocessing.

## Regenerate after changing a recipe

Edit the source YAML, then regenerate all encoder scripts:

```bash
for cluster in local slurm slurm_bio runai; do
  python scripts/launch.py experiments/encoder --generate --cluster "$cluster" || break
done
```

To generate one custom run (including on a machine with Python for a Run:ai submit host):

```bash
python scripts/launch.py experiments/encoder/conv_mlp.yaml \
  --cluster runai --generate --set seed=142
# Writes experiments/generated/encoder/conv_mlp_s142.runai.sh
```
