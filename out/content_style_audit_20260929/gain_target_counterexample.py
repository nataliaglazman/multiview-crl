"""Exact scalar example of the cross-subject target mismatch; no model training.

Assume perfectly separated, fixed anatomy of intensity 1 and independent gain
values in {0.7, 1.0, 1.3}. The intended swap uses the donor's gain. Scoring that
swap against the recipient's gain rewards ignoring the donor instead.
"""

import json
from fractions import Fraction


def mean(values):
    return sum(values, Fraction(0)) / len(values)


gains = [Fraction(7, 10), Fraction(1), Fraction(13, 10)]
pairs = [(recipient, donor) for recipient in gains for donor in gains]
median_gain = Fraction(1)

wrong_target_correct_transfer = mean([abs(donor - recipient) for recipient, donor in pairs])
wrong_target_ignore_donor = mean([abs(median_gain - recipient) for recipient, donor in pairs])
correct_target_correct_transfer = mean([abs(donor - donor) for recipient, donor in pairs])
correct_target_ignore_donor = mean([abs(median_gain - donor) for recipient, donor in pairs])

assert wrong_target_ignore_donor < wrong_target_correct_transfer
assert correct_target_correct_transfer < correct_target_ignore_donor

print(
    json.dumps(
        {
            "scope": "Analytical toy example; not measurements of the VQ-VAE checkpoint",
            "gain_values": [float(g) for g in gains],
            "all_independent_recipient_donor_pairs": len(pairs),
            "recipient_original_target": {
                "correct_donor_gain_transfer_mae": float(wrong_target_correct_transfer),
                "ignore_donor_use_median_gain_mae": float(wrong_target_ignore_donor),
            },
            "mixed_anatomy_donor_style_target": {
                "correct_donor_gain_transfer_mae": float(correct_target_correct_transfer),
                "ignore_donor_use_median_gain_mae": float(correct_target_ignore_donor),
            },
        },
        indent=2,
    )
)
