# Encoder-only ADNI modality diagnostic

This tests whether a checkpoint's content modality leakage is explained by a
constant offset **in L2-normalized content space**. It loads the encoder architecture
from `settings.json`, the requested weights (default `model_best.pt`), and the
original validation subjects. It checks their identity/order against `split.json`.
It does not train the encoder or load the test split.

## Run

From the repository root, inside the training environment:

```bash
python -m eval.adni.encoder_modality_gap \
    --run-dir results/encoder_adni/runs/encoder_adni_conv_mlp_layernorm_s42 \
    --checkpoint model_best.pt --device cuda --batch-size 4
```

On the Run:ai submission host, after syncing this code to the NFS checkout:

```bash
runai training standard submit adni-modality-gap-$(date +%s) \
    --project nglazman \
    --image aicregistry:5000/nglazman:multiview-crl \
    --run-as-user --large-shm --node-type A100 --gpu-devices-request 1 \
    --cpu-core-request 4 --cpu-core-limit 8 \
    --cpu-memory-request 32G --cpu-memory-limit 64G \
    --host-path path=/nfs,mount=/nfs,readwrite \
    --command -- bash -c 'set -euo pipefail
        cd /nfs/home/nglazman/crl-2/multiview-crl
        export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
        export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
        python -m eval.adni.encoder_modality_gap \
            --run-dir results/encoder_adni/runs/encoder_adni_conv_mlp_layernorm_s42 \
            --checkpoint model_best.pt --device cuda --batch-size 4'
```

Change the run directory for other model IDs. To compare with the actual untrained
floor, repeat with `--checkpoint model_init.pt`; both runs use identical probe folds
at the default `--seed 1729`. Use a finished/stable checkpoint file while extracting.
`--dataroot`, `--labels-path`, `--masks-dir`, and `--cache-dir` can override saved
cluster paths when moving a run. Subject membership must still match the saved split.

Each invocation creates a new `modality_gap_<timestamp>` directory within the run:

- `report.json`: geometry, four probe scores, individual fold scores, fitted
  modality means, protocol, settings, and checkpoint SHA-256.
- `features.npz`: unnormalized content vectors from both views and subject IDs.
- `predictions.csv`: one row per scan/view, with fold and all four held-out predictions.

Re-score saved features on a machine with NumPy, scikit-learn and threadpoolctl;
this path does not import PyTorch or read MRI volumes:

```bash
python -m eval.adni.encoder_modality_gap \
    --features /path/to/modality_gap_TIMESTAMP/features.npz \
    --out-dir /path/to/new_report
```

An existing output directory is refused. The tool fails on missing/mismatched
subjects or non-finite features rather than substituting a chance score.

## What is measured

The content block is the first `content_channels` global encoder outputs, before
any contrastive projector, matching the training-time modality probe. Each vector
is first L2-normalized. Zero vectors remain zero and are counted explicitly;
cosine averages exclude pairs containing them.

The descriptive geometry reports:

- **Centroid distance:** `||mean(T1) - mean(FLAIR)||` after normalization.
- **Distance relative to within-modality spread:** that distance divided by the
  pooled RMS distance from each modality's own centroid.
- **Paired cosine:** mean and standard deviation of the same subject's T1/FLAIR
  cosine similarity.
- **Mean offset fraction:** squared centroid distance divided by mean paired
  squared distance. This describes how much of the paired discrepancy is a mean
  difference; it is not a fraction of modality information explained.

Geometry uses the whole validation cohort descriptively. None of its fitted
quantities is supplied to the cross-validation probes.

The default three folds split **subjects**, keeping all scans and both views of a
subject together. Each fold compares:

| Condition | Preprocessing before the probe |
| --- | --- |
| `normalized` | L2 normalization, then pooled StandardScaler fitted on training rows |
| `mean_centered` | L2 normalization, subtract each modality's training-fold mean, then pooled StandardScaler fitted on training rows |

The held-out vectors use the training-fold means and scaler. There is no second
L2 normalization after centering, since that would introduce an additional
nonlinear transformation. Both conditions use fixed logistic regression (`C=1`)
and an RBF SVM (`C=1`, `gamma=scale`). Probe hyperparameters are not tuned on this
cohort. Accuracy is pooled across held-out predictions; chance is 0.5 because the
two modalities have equal counts. Per-fold scores are included for inspection.
Channels with training-fold standard deviation at most `1e-12` are zeroed in both
partitions before scaling, preventing floating-point residue after centering from
being amplified into apparent modality information. These channels are recorded
per fold and condition.

## Interpretation

**Read the centered RBF probe alongside the linear probe.** Removing each class
mean makes a regularized logistic probe with an intercept uninformative on its
training data by construction. Its chance score alone cannot distinguish a pure
offset from more complicated modality differences. A nonlinear probe can detect
residual shape/covariance differences.

- High original accuracy, followed by both centered probes near chance, is
  consistent with a removable normalized-space mean offset at this probe capacity
  and sample size. It does not prove equal distributions.
- High centered RBF accuracy detects differences beyond a constant mean offset.
- High paired cosine can coexist with perfect modality classification: a tiny
  consistent direction can identify modality while subject matches remain close.

Centering deliberately uses known modality labels, including the held-out scan's
modality, to select a **training-fitted** mean. The centered result describes this
intervention; it must not be reported as invariance learned by the encoder. It also
does not test diagnosis/anatomy preservation. A constant offset before normalization
need not be constant after normalization, so conclusions refer to normalized space.

The diagnostic does not change the training-time probe. Its scores may differ
because the original probe fits StandardScaler before CV; this diagnostic fits it
within every fold and explicitly groups subjects.

## Verification

```bash
python -m unittest discover -s tests -p test_encoder_modality_gap.py -v
```

Tests cover a tiny planted offset, a residual nonlinear modality difference with
equal population means, identical/collapsed features, repeated-subject grouping,
training-only centering, saved-feature replay, and frozen extraction from a small
fake ADNI tree with validation membership verified against the saved run.
