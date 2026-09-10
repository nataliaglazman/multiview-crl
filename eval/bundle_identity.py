"""Row identity for the shared evaluation bundles.

Two feature bundles may be compared only when they describe *the same evaluation rows*:
the same factor draws, in the same order, from the same generator settings.  Nothing else
about a bundle establishes that.  Matching sample counts do not -- two runs of the
generator at different ``synthetic_seed`` produce 2000 rows each and share not one of
them.  Matching directory names do not either, which is the failure this module exists to
stop: a DINO bundle and a VQ-VAE bundle sitting in one results tree, differenced factor by
factor, describing different brains.

The identity is three fields, written into every bundle's ``meta`` by whichever script
produced it:

``factor_digest``
    SHA-256 over the ground-truth factor arrays as they are stored (float32, C order),
    in a fixed key order.  This is the strong check: it is the actual evaluation target,
    so two bundles that agree here were scored against identical labels whatever else
    differs.
``generator_digest``
    SHA-256 over the generator settings dict.  Weaker but more informative when the
    factors differ, because it separates "different settings" from "same settings,
    different draw".
``n_rows``
    Row count, kept separately so a length mismatch reports as a length mismatch rather
    than as an opaque digest difference.

Digests are taken on the float32 view because that is the dtype the writers save, so a
digest computed before writing equals one recomputed after loading.  They deliberately do
NOT cover the features: a bundle's whole purpose is to carry different features for the
same rows.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

#: Ground-truth arrays that define a row, in the order they enter the digest.  Adding a
#: key here changes every digest, so it is append-only in practice.
FACTOR_KEYS = ("z_content", "z_style_v1", "z_style_v2", "causal_adj")

#: ``meta`` keys this module owns.
IDENTITY_KEYS = ("factor_digest", "generator_digest", "n_rows")


def _digest_array(hasher, name, array):
    """Feed one array to ``hasher``, name and shape included.

    The shape goes in because the bytes alone do not distinguish an ``(N, 2K)`` array
    from the ``(2N, K)`` one with the same buffer, and a factor set that changed width is
    exactly the kind of mismatch worth catching.
    """
    values = np.ascontiguousarray(np.asarray(array), dtype=np.float32)
    hasher.update(name.encode("utf-8"))
    hasher.update(repr(values.shape).encode("utf-8"))
    hasher.update(values.tobytes())


def factor_digest(latents):
    """SHA-256 over the ground-truth arrays in ``latents``, by :data:`FACTOR_KEYS` order.

    Missing keys are skipped rather than defaulted, so a single-view bundle and a
    two-view one built from the same draw do not silently digest to the same value: the
    key name is hashed alongside its bytes.
    """
    hasher = hashlib.sha256()
    for key in FACTOR_KEYS:
        value = latents.get(key)
        if value is None:
            continue
        value = np.asarray(value)
        # The dataloader's collate stacks one adjacency per sample, all of them the same
        # SCM, and the writers store only the first. Reduce here too, so the digest taken
        # before writing equals the one recomputed after loading.
        if key == "causal_adj" and value.ndim == 3:
            value = value[0]
        _digest_array(hasher, key, value)
    return hasher.hexdigest()


def generator_digest(settings):
    """SHA-256 over the generator settings, key-sorted so dict order cannot move it."""
    payload = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def identity_record(latents, settings, n_rows):
    """The three ``meta`` fields describing which rows a bundle holds."""
    return {
        "factor_digest": factor_digest(latents),
        "generator_digest": generator_digest(settings),
        "n_rows": int(n_rows),
    }


def read_identity(meta, latents=None, n_rows=None):
    """A bundle's identity, recomputed from its arrays when its ``meta`` predates this.

    Bundles written before the identity fields existed still compare correctly, because
    the digest is a pure function of the stored factor arrays.  ``generator_digest`` is
    not recoverable that way and stays ``None``, which the comparison treats as unknown
    rather than as a mismatch.
    """
    record = {key: meta.get(key) for key in IDENTITY_KEYS}
    if record["factor_digest"] is None and latents is not None:
        record["factor_digest"] = factor_digest(latents)
    if record["n_rows"] is None and n_rows is not None:
        record["n_rows"] = int(n_rows)
    return record


def compare(records):
    """Differences among ``{label: identity_record}``. Empty list means comparable.

    Reported worst-first: a row-count difference explains a digest difference, so listing
    the digest too would be noise.
    """
    labels = list(records)
    if len(labels) < 2:
        return []
    problems = []

    counts = {label: records[label].get("n_rows") for label in labels}
    if len({c for c in counts.values() if c is not None}) > 1:
        detail = ", ".join(f"{label}={counts[label]}" for label in labels)
        return [f"different row counts ({detail}); the bundles were built with different --num-samples"]

    digests = {label: records[label].get("factor_digest") for label in labels}
    known = {d for d in digests.values() if d is not None}
    if len(known) > 1:
        detail = ", ".join(f"{label}={(digests[label] or 'unknown')[:12]}" for label in labels)
        problems.append(
            f"different ground-truth factors ({detail}); same row count but not the same rows, "
            "so a per-factor difference between these bundles is not a model difference"
        )
        generators = {records[label].get("generator_digest") for label in labels}
        if len({g for g in generators if g is not None}) > 1:
            problems.append(
                "their generator settings also differ -- rebuild both with the same --run-dir "
                "(or the same explicit generator flags) and the same --num-samples"
            )
        elif None not in generators:
            problems.append(
                "their generator settings agree, so this is a different draw of the same "
                "generator -- check --seed / --synthetic-seed and the split"
            )
    elif not known:
        problems.append("no bundle carries a factor digest, so row identity could not be verified")
    return problems


def require_same_rows(records):
    """Raise ``ValueError`` unless every bundle in ``records`` describes the same rows."""
    problems = compare(records)
    if problems:
        raise ValueError("Bundles are not row-aligned:\n  - " + "\n  - ".join(problems))
