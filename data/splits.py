"""Subject-level train/val/test splitting for the real (ADNI) datasets.

The synthetic generator gets disjoint splits for free — it draws each split from
its own seed offset.  Real data does not: ``load_data`` resolves every row of the
labels CSV, so without an explicit split the train, val and test datasets are the
same subjects and every validation number is a training number.

This module is deliberately free of torch/MONAI imports so the split can be
computed (and self-tested) without the training environment — ``scripts/
preflight_adni.py`` uses it to report split sizes before a job is submitted.

Two properties matter here:

* **Subject-level.**  ADNI subjects can contribute more than one row.  Splitting
  on rows would put the same brain in train and val, and content representations
  are precisely what would leak.
* **Stratified by label.**  With three diagnosis groups and a few hundred
  subjects, an unstratified draw routinely leaves a val split whose group
  proportions differ enough to move the diagnosis probe on its own.
"""

MODES = ("train", "val", "test")


def subject_level_split(items, val_frac=0.0, test_frac=0.0, seed=0):
    """Split ``items`` into disjoint train/val/test index lists.

    Args:
        items: List of dicts as returned by ``utils.utils.load_data`` — each must
            carry ``"subject"``, and ``"label"`` when stratification is wanted.
        val_frac: Fraction of *subjects* held out for validation.
        test_frac: Fraction of *subjects* held out for test.
        seed: Seed for the deterministic subject shuffle.

    Returns:
        dict: ``{"train": [...], "val": [...], "test": [...]}`` of positional
        indices into ``items``, each sorted ascending.

    When both fractions are zero every mode gets *all* indices.  That is the
    historical behaviour (train == val == test) and is kept as the default so
    existing runs and their preprocessed caches are unaffected; opt in to a real
    split with ``--val-frac`` / ``--test-frac``.
    """
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    if not 0.0 <= test_frac < 1.0:
        raise ValueError(f"test_frac must be in [0, 1), got {test_frac}")
    if val_frac + test_frac >= 1.0:
        raise ValueError(f"val_frac + test_frac must be < 1, got {val_frac + test_frac}")

    all_indices = list(range(len(items)))
    if val_frac == 0.0 and test_frac == 0.0:
        return {mode: list(all_indices) for mode in MODES}

    # subject -> positional indices, in first-appearance order so the result is
    # independent of dict iteration order.
    by_subject = {}
    for i, item in enumerate(items):
        by_subject.setdefault(str(item["subject"]), []).append(i)

    # Stratification bucket per subject: the label of its first row.  A subject
    # with inconsistent labels across rows is bucketed by that first label rather
    # than dropped — losing scans silently would be worse than a slight
    # imbalance, and preflight reports the inconsistency.
    bucket_of = {}
    for subject, idxs in by_subject.items():
        bucket_of[subject] = items[idxs[0]].get("label", 0)

    buckets = {}
    for subject in sorted(by_subject):
        buckets.setdefault(bucket_of[subject], []).append(subject)

    assigned = {mode: [] for mode in MODES}
    for label in sorted(buckets, key=lambda x: (str(type(x)), x)):
        subjects = _shuffled(buckets[label], seed, label)
        n = len(subjects)
        # Reserve at least one subject for train before anything is held out, so
        # a small class can never empty the training set.
        n_test = _hold_out_count(n, test_frac, reserved=1)
        n_val = _hold_out_count(n - n_test, val_frac, reserved=1)
        cursor = 0
        for mode, count in (("test", n_test), ("val", n_val)):
            for subject in subjects[cursor : cursor + count]:
                assigned[mode].extend(by_subject[subject])
            cursor += count
        for subject in subjects[cursor:]:
            assigned["train"].extend(by_subject[subject])

    return {mode: sorted(idxs) for mode, idxs in assigned.items()}


def _hold_out_count(n, frac, reserved=0):
    """How many of ``n`` subjects to hold out at ``frac``, keeping ``reserved`` back.

    Rounds to nearest, but promotes a rounded-down zero to one whenever the class
    is large enough to spare a subject.  Without that, a 12-subject class at
    ``val_frac=0.15`` contributes nothing to val and the split silently stops
    being stratified.
    """
    if frac <= 0.0 or n <= reserved:
        return 0
    count = int(n * frac + 0.5)
    if count == 0:
        count = 1
    return min(count, n - reserved)


def _shuffled(subjects, seed, label):
    """Deterministic shuffle of ``subjects``, independent of the other classes.

    Seeding per class means adding or removing a class does not reshuffle the
    others, so a split stays comparable across CSV revisions that only touch one
    group.
    """
    import random

    out = list(subjects)
    random.Random(f"{seed}:{label}").shuffle(out)
    return out


def split_summary(items, split, label_names=None):
    """Human-readable per-split subject/scan/group counts.

    Returns a list of lines; used by preflight and logged at the start of a run
    so the split that produced a set of numbers is recoverable from the log.
    """
    lines = []
    disjoint = _is_disjoint(split)
    for mode in MODES:
        idxs = split[mode]
        subjects = {str(items[i]["subject"]) for i in idxs}
        by_label = {}
        for i in idxs:
            key = items[i].get("label", 0)
            by_label[key] = by_label.get(key, 0) + 1
        groups = ", ".join(
            f"{_label_name(k, label_names)}={by_label[k]}" for k in sorted(by_label, key=lambda x: str(x))
        )
        lines.append(
            f"  {mode:<5} {len(subjects):>4} subjects, {len(idxs):>4} scans" + (f"  ({groups})" if groups else "")
        )
    if not disjoint:
        lines.append("  NOTE: splits overlap — no split configured (train == val == test).")
    return lines


def _label_name(key, label_names):
    if label_names and key in label_names:
        return str(label_names[key])
    return str(key)


def _is_disjoint(split):
    seen = set()
    for mode in MODES:
        idxs = set(split[mode])
        if seen & idxs:
            return False
        seen |= idxs
    return True


def _self_test():
    """Torch-free checks: ``python -m data.splits --self-test``."""
    failures = []

    def check(name, cond):
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        if not cond:
            failures.append(name)

    # Two scans for some subjects, three diagnosis groups.
    items = []
    for s in range(60):
        label = s % 3
        n_scans = 2 if s % 7 == 0 else 1
        for _ in range(n_scans):
            items.append({"subject": f"S{s:03d}", "label": label})

    print("default (no fractions) reproduces historical behaviour:")
    split = subject_level_split(items)
    check("every mode gets all indices", all(len(split[m]) == len(items) for m in MODES))

    print("\nsplit at val=0.2 test=0.1:")
    split = subject_level_split(items, val_frac=0.2, test_frac=0.1, seed=0)
    check("modes are disjoint", _is_disjoint(split))
    check("covers every index", sorted(sum(split.values(), [])) == list(range(len(items))))

    subs = {m: {items[i]["subject"] for i in split[m]} for m in MODES}
    check("no subject spans two splits", not (subs["train"] & subs["val"]) and not (subs["train"] & subs["test"]))
    check("val is roughly 20% of subjects", 0.15 <= len(subs["val"]) / 60 <= 0.27)
    check("test is roughly 10% of subjects", 0.05 <= len(subs["test"]) / 60 <= 0.18)

    labels_in = {m: {items[i]["label"] for i in split[m]} for m in MODES}
    check("all three groups present in val", labels_in["val"] == {0, 1, 2})
    check("all three groups present in test", labels_in["test"] == {0, 1, 2})

    print("\ndeterminism:")
    again = subject_level_split(items, val_frac=0.2, test_frac=0.1, seed=0)
    check("same seed gives same split", again == split)
    other = subject_level_split(items, val_frac=0.2, test_frac=0.1, seed=1)
    check("different seed gives different split", other != split)

    print("\nedge cases:")
    tiny = [{"subject": "A", "label": 0}, {"subject": "B", "label": 0}]
    tiny_split = subject_level_split(tiny, val_frac=0.5, test_frac=0.0, seed=0)
    check("2 subjects at 50% keeps one for train", len(tiny_split["train"]) == 1 and len(tiny_split["val"]) == 1)

    single = [{"subject": "A", "label": 0}]
    single_split = subject_level_split(single, val_frac=0.3, test_frac=0.3, seed=0)
    check("1 subject stays in train", single_split["train"] == [0] and not single_split["val"])

    rare = [{"subject": f"S{i}", "label": 0} for i in range(30)] + [{"subject": "R0", "label": 1}]
    rare_split = subject_level_split(rare, val_frac=0.2, test_frac=0.1, seed=0)
    check("a 1-subject class is not held out", 30 in rare_split["train"])

    for bad in ({"val_frac": 1.0}, {"test_frac": -0.1}, {"val_frac": 0.6, "test_frac": 0.5}):
        try:
            subject_level_split(items, **bad)
            check(f"rejects {bad}", False)
        except ValueError:
            check(f"rejects {bad}", True)

    print("\nsummary rendering:")
    for line in split_summary(items, split, label_names={0: "CN", 1: "MCI", 2: "AD"}):
        print(line)

    print("\n" + ("ALL PASSED" if not failures else f"{len(failures)} FAILED: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        sys.exit(_self_test())
    print("Nothing to do. Run with --self-test.")
