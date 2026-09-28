"""Shared Barlow Twins arm construction for training and frozen gradient audits.

Training and ``eval.loss_gradient_audit`` both build their loss callables here, so the
audited objective cannot drift from the trained one.
"""


def make_barlow_loss_functions(
    args, *, states=None, correlations=None, capture_components=False, loss_impl=None, stats_impl=None
):
    """Return ``(plain, patch)`` loss callables with training's signature.

    ``states`` holds one correlation-EMA dict per arm ("global", "patch", "gap") and must
    persist across calls, as training's did. ``correlations`` receives each arm's
    instantaneous correlation matrices; ``capture_components`` attaches weighted per-term
    tensors as ``._loss_components``. Both are audit-only and leave the loss unchanged.
    """
    if loss_impl is None:
        from training.losses import barlow_twins_loss, stats_pool

        loss_impl, stats_impl = barlow_twins_loss, stats_pool
    states = states if states is not None else {}

    def setting(name, default):
        value = getattr(args, name, None)
        return default if value is None else value

    def arm(hz, indices, subsets, mask, name):
        gap = name == "gap"
        options = dict(
            lambd=setting("bt_lambda", 0.005),
            sim_coeff=setting("bt_sim_coeff", 0.0),
            std_coeff=setting("bt_std_coeff", 0.0),
        )
        if gap:
            for key, flag in (("lambd", "lambda"), ("sim_coeff", "sim_coeff"), ("std_coeff", "std_coeff")):
                options[key] = setting(f"bt_gap_{flag}", options[key])
        loss = loss_impl(
            hz,
            estimated_content_indices=indices,
            subsets=subsets,
            soft_content_mask=mask,
            center_mode=setting("patch_center_mode", "none") if name == "patch" else "none",
            patch_stat=setting("bt_patch_stat", "fold"),
            sim_normalize=setting("bt_sim_normalize", False),
            # Never whiten the patch fold: its rows are (subject, position) pairs, not the
            # aligned units, and at d~768 a d x d covariance is not estimable per batch.
            sim_whiten=setting("bt_sim_whiten", False) if name != "patch" else False,
            sim_whiten_eps=setting("bt_sim_whiten_eps", 1e-3),
            corr_ema=states.setdefault(name, {}),
            corr_ema_decay=setting("bt_corr_ema", 0.0),
            normalize_terms=setting("bt_normalize_terms", False),
            capture_components=capture_components,
            **options,
        )
        if correlations is not None:
            correlations[name] = getattr(loss, "_bt_correlations", {})
        return loss

    def combine(parts):
        # Arithmetic drops tensor attributes, so the diagnostics are re-attached; losing them
        # would silently blank every Contrastive/* curve. GAP diagnostics log as gap_*.
        total = sum(weight * loss for _, weight, loss in parts)
        diagnostics, components = {}, {}
        for name, weight, loss in parts:
            prefix = "gap_" if name == "gap" else ""
            diagnostics.update({prefix + k: v for k, v in getattr(loss, "_contrastive_diag", {}).items()})
            if capture_components:
                components.update({f"{name}/{k}": weight * v for k, v in loss._loss_components.items()})
        total._contrastive_diag = diagnostics
        if capture_components:
            total._loss_components = components
        return total

    def plain(hz, estimated_content_indices, subsets, soft_content_mask=None):
        loss = arm(hz, estimated_content_indices, subsets, soft_content_mask, "global")
        return combine([("global", 1.0, loss)])

    def patch(hz, estimated_content_indices, subsets, soft_content_mask=None):
        parts = [
            (
                "patch",
                setting("bt_patch_weight", 1.0),
                arm(hz, estimated_content_indices, subsets, soft_content_mask, "patch"),
            )
        ]
        gap_weight = setting("bt_gap_weight", 0.0)
        # The patch fold's cross-covariance is Cov_subject + Cov_interaction, and the
        # interaction dominates on registered volumes. Pooling over positions recovers the
        # subject term exactly, so the GAP arm is the one whose rows are SUBJECTS.
        # bt_patch_weight 0 with this arm on gives a GAP-only objective.
        if gap_weight > 0 and hz.ndim == 4:
            # Both poolings give one row per subject. The mean keeps ~1/P of a localised
            # factor; stats also keeps each channel's spread and extremes (measured:
            # ventricle_size content R^2 0.097 at gap, 0.406 at stats).
            if setting("bt_gap_pooling", "gap") == "stats":
                pooled, indices, mask = stats_impl(hz, estimated_content_indices, soft_content_mask)
            else:
                pooled, indices, mask = hz.mean(-1), estimated_content_indices, soft_content_mask
            parts.append(("gap", gap_weight, arm(pooled, indices, subsets, mask, "gap")))
        return combine(parts)

    return plain, patch
