"""Decode optional ADNI categorical probe targets; -1 denotes unavailable labels."""

DEMOGRAPHIC_COLUMNS = {"gender": "PTGENDER", "race": "PTRACCAT"}


def decode_demographics(row):
    targets = {}
    for target, column in DEMOGRAPHIC_COLUMNS.items():
        if column not in row:
            continue
        try:
            values = [float(part.strip()) for part in str(row[column]).split("|")]
            codes = {int(value) for value in values if value.is_integer()}
            valid = len(codes) > 0 and all(value.is_integer() for value in values)
        except (ValueError, OverflowError):
            codes, valid = set(), False
        allowed = {1, 2} if target == "gender" else {1, 2, 3, 4, 5, 6, 8, 9}
        if not valid or not codes <= allowed or (target == "gender" and len(codes) != 1):
            targets[target] = -1
        else:
            # ADNI4 uses pipe-separated selections; harmonize multiple selections
            # to the historical "more than one race" code, retaining single codes.
            targets[target] = 6 if len(codes) > 1 else next(iter(codes))
    return targets
