"""
Collection of datasets.
"""

import os
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch

from utils.utils import load_data


class MultiviewDataset(torch.utils.data.Dataset):
    FACTORS = None
    DISCRETE_FACTORS = None
    FACTOR_SIZES = None
    LATENT_SPACES = None

    mean_per_channel = [0.0] * 3
    std_per_channel = [1.0] * 3

    @abstractmethod
    def __getview__(self, item):
        raise NotImplementedError

    @abstractmethod
    def __get_augmented_view__(self, idx, z, change_list):
        raise NotImplementedError

    @abstractmethod
    def sample(self, size, random_state=None):
        raise NotImplementedError


class MyCustomDataset(MultiviewDataset):
    """ADNI dataset with T1 and T2 as two views.

    When ``cache=True`` all volumes are loaded, preprocessed, and stored in RAM
    during ``__init__``.  Subsequent ``__getitem__`` calls only apply stochastic
    augmentations (``RandAffined``, ``RandShiftIntensityd``) on the cached
    tensors, completely avoiding repeated NIfTI disk I/O and resampling.
    """

    # Not used for 3D medical images, but required by parent class
    mean_per_channel = [0.0]
    std_per_channel = [1.0]

    # Minimal factors definition (not used for training, only for compatibility)
    FACTORS = {"image": {0: "view"}}
    DISCRETE_FACTORS = {"image": {}}
    LATENT_SPACES = {"image": {}}

    def __init__(
        self,
        data_dir: str,
        mode="train",
        transform=None,
        spacing=2.0,
        crop_margin=0,
        spatial_size=None,
        cache=False,
        cache_dir: str | None = None,
        labels_path: str | None = None,
        masks_dir: str | None = None,
        asymmetric_aug: bool = False,
        shared_brain_mask: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.mode = mode
        self.data_dir = data_dir
        self.spacing = spacing
        self.crop_margin = crop_margin
        self.spatial_size = tuple(spatial_size) if spatial_size is not None else None
        self.masks_dir = masks_dir
        self.asymmetric_aug = asymmetric_aug
        self.shared_brain_mask = shared_brain_mask

        # Load CSV and build item list
        if labels_path is None:
            raise ValueError("labels_path is required. Pass --labels-path <path> on the command line.")
        df = pd.read_csv(labels_path)
        label_values = sorted(df["Group"].unique())
        label_map = {v: i for i, v in enumerate(label_values)}
        self.label_map = label_map  # group_name → int

        # Load data using utils.load_data
        self.items, missing = load_data(df, data_dir, label_map, masks_dir=masks_dir)
        self.num_samples = len(self.items)

        # All items must carry brain-mask paths to enable the mask pipeline —
        # if any item is missing masks, fall back to the thresholding path so
        # every sample follows the same transform sequence.
        self.masks_from_disk = all("mask_image" in it and "mask_z_image" in it for it in self.items)

        # Get MONAI transforms with specified spacing and cropping
        from utils.utils import transforms as get_transforms

        train_transforms, val_transforms = get_transforms(
            spacing=self.spacing,
            crop_margin=self.crop_margin,
            spatial_size=self.spatial_size,
            masks_from_disk=self.masks_from_disk,
            asymmetric_aug=self.asymmetric_aug,
            shared_brain_mask=self.shared_brain_mask,
        )

        if cache:
            self._build_cache(val_transforms, cache_dir=cache_dir)
            # Augmentation-only pipeline for training (parameters mirror
            # those in utils.utils.transforms — keep in sync).
            if mode == "train":
                from monai.transforms import (
                    Compose,
                    RandAdjustContrastd,
                    RandAffined,
                    RandBiasFieldd,
                    RandGaussianNoised,
                    RandGaussianSmoothd,
                    RandScaleIntensityd,
                    RandShiftIntensityd,
                )

                from utils.utils import ApplyBrainMaskd

                # Affine must apply the same transform to image and mask so
                # they stay spatially aligned.
                aug_list = [
                    RandAffined(
                        keys=["image_t1", "image_t2", "mask_t1", "mask_t2"],
                        mode=["bilinear", "bilinear", "nearest", "nearest"],
                        rotate_range=[-0.05, 0.05],
                        shear_range=[0.001, 0.05],
                        scale_range=[0, 0.05],
                        padding_mode="zeros",
                        prob=0.5,
                    )
                ]
                if self.asymmetric_aug:
                    # Independent intensity perturbations per view.
                    for view_key in ("image_t1", "image_t2"):
                        aug_list.extend(
                            [
                                RandShiftIntensityd(keys=[view_key], offsets=(-0.1, 0.1), prob=0.5),
                                RandScaleIntensityd(keys=[view_key], factors=0.1, prob=0.5),
                                RandBiasFieldd(keys=[view_key], coeff_range=(0.0, 0.1), prob=0.3),
                                RandAdjustContrastd(keys=[view_key], gamma=(0.7, 1.5), prob=0.3),
                                RandGaussianNoised(keys=[view_key], std=0.05, prob=0.3),
                                RandGaussianSmoothd(
                                    keys=[view_key],
                                    sigma_x=(0.25, 1.0),
                                    sigma_y=(0.25, 1.0),
                                    sigma_z=(0.25, 1.0),
                                    prob=0.2,
                                ),
                            ]
                        )
                    aug_list.append(
                        ApplyBrainMaskd(
                            keys=["image_t1", "image_t2"],
                            mask_keys=["mask_t1", "mask_t2"],
                            threshold=0.5,
                        )
                    )
                else:
                    aug_list.append(RandShiftIntensityd(keys=["image_t1", "image_t2"], offsets=(-0.1, 0.1), prob=0.2))
                self._aug_transform = Compose(aug_list)
            else:
                self._aug_transform = None
            self.monai_transform = None  # not used when cached
        else:
            self._cache = None
            self._aug_transform = None
            self.monai_transform = train_transforms if mode == "train" else val_transforms

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _transform_one(args):
        """Process a single subject through the deterministic transform pipeline.

        This is a module-level-compatible static method so it can be pickled by
        multiprocessing workers.
        """
        idx, item, deterministic_transform = args
        data_dict = {
            "image_t1": item["image"],
            "image_t2": item["z_image"],
            "label": item["label"],
        }
        if "mask_image" in item and "mask_z_image" in item:
            data_dict["mask_t1"] = item["mask_image"]
            data_dict["mask_t2"] = item["mask_z_image"]
        transformed = deterministic_transform(data_dict)
        cached = {
            "image_t1": transformed["image_t1"],
            "image_t2": transformed["image_t2"],
            "label": transformed["label"],
            "mask_t1": transformed["mask_t1"],
            "mask_t2": transformed["mask_t2"],
        }
        return idx, cached

    @staticmethod
    def _transform_and_save(args):
        """Process a single subject and persist the result to disk.

        Used by the persistent-cache path.  Each sample is saved as a
        separate ``.pt`` file so that partial caches are resumable and
        individual files can be memory-mapped on load.
        """
        import torch as _torch

        idx, item, deterministic_transform, save_path = args
        # Skip if already cached on disk from a previous (possibly interrupted) run
        if os.path.exists(save_path):
            return idx, save_path
        # Ensure parent dir exists — NFS workers may not see it immediately
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data_dict = {
            "image_t1": item["image"],
            "image_t2": item["z_image"],
            "label": item["label"],
        }
        if "mask_image" in item and "mask_z_image" in item:
            data_dict["mask_t1"] = item["mask_image"]
            data_dict["mask_t2"] = item["mask_z_image"]
        transformed = deterministic_transform(data_dict)
        # Convert to plain torch tensors — MONAI MetaTensors carry numpy
        # internals that torch.load(weights_only=True) rejects.
        t1 = _torch.as_tensor(transformed["image_t1"]).clone().contiguous()
        t2 = _torch.as_tensor(transformed["image_t2"]).clone().contiguous()
        m1 = _torch.as_tensor(transformed["mask_t1"]).clone().contiguous()
        m2 = _torch.as_tensor(transformed["mask_t2"]).clone().contiguous()
        lbl = int(transformed["label"]) if not isinstance(transformed["label"], int) else transformed["label"]
        cached = {
            "image_t1": t1,
            "image_t2": t2,
            "mask_t1": m1,
            "mask_t2": m2,
            "label": lbl,
        }
        # Atomic write with memory staging:
        # PyTorch's internal ZIP stream writer can fail with "unexpected pos"
        # when seeking back-and-forth directly on a mounted NFS drive.
        # We write the dictionary to a RAM buffer first, then flush the
        # finalized bytes sequentially to the NFS to guarantee stability.
        import io
        import uuid

        buffer = io.BytesIO()
        _torch.save(cached, buffer)

        nfs_tmp_path = save_path + f".tmp.{uuid.uuid4().hex[:8]}"
        with open(nfs_tmp_path, "wb") as f:
            f.write(buffer.getvalue())

        # 3. Atomic finalize
        os.replace(nfs_tmp_path, save_path)
        return idx, save_path

    # ------------------------------------------------------------------
    # Cache fingerprinting
    # ------------------------------------------------------------------

    def _cache_fingerprint(self) -> str:
        """Compute a hex digest that uniquely identifies this dataset config.

        The fingerprint covers transform parameters and the ordered set of
        source file paths, so the disk cache is automatically invalidated when
        any of these change.
        """
        import hashlib

        h = hashlib.sha256()
        h.update(f"spacing={self.spacing}".encode())
        h.update(f"crop_margin={self.crop_margin}".encode())
        h.update(f"spatial_size={self.spatial_size}".encode())
        h.update(f"masks_from_disk={self.masks_from_disk}".encode())
        h.update(f"shared_brain_mask={self.shared_brain_mask}".encode())
        for item in self.items:
            h.update(item["image"].encode())
            h.update(item["z_image"].encode())
            if "mask_image" in item:
                h.update(item["mask_image"].encode())
                h.update(item["mask_z_image"].encode())
        return h.hexdigest()[:16]

    # ------------------------------------------------------------------
    # Cache building
    # ------------------------------------------------------------------

    def _build_cache(self, deterministic_transform, num_workers=None, cache_dir=None):
        """Load preprocessed volumes into ``self._cache``.

        If ``cache_dir`` is provided (or defaults to ``<data_dir>/.cache``),
        the method first checks for a valid persistent disk cache:

        * **Hit** — every per-sample ``.pt`` file exists and the fingerprint
          matches.  Tensors are memory-mapped into RAM (PyTorch ≥ 2.1) so the
          OS pages data on demand.  Startup drops from minutes to seconds.
        * **Partial hit** — some ``.pt`` files are present (e.g. from a
          previously interrupted run).  Only the missing samples are processed
          and written; existing files are skipped.
        * **Miss** — samples are processed in parallel and saved to disk for
          next time.

        When ``cache_dir is None`` the behaviour is identical to the original
        RAM-only cache (no disk I/O).

        Args:
            deterministic_transform: MONAI ``Compose`` of deterministic transforms.
            num_workers: Parallel workers (default ``min(cpu_count, 8)``).
            cache_dir: Directory for persistent ``.pt`` files.  ``None`` disables
                       disk persistence and caches in RAM only.
        """
        import logging
        import sys
        from pathlib import Path

        logger = logging.getLogger("multiview_crl")

        if num_workers is None:
            num_workers = min(os.cpu_count() or 1, 8)

        # ------------------------------------------------------------------
        # Persistent disk cache path
        # ------------------------------------------------------------------
        if cache_dir is not None:
            fingerprint = self._cache_fingerprint()
            cache_root = Path(cache_dir) / f"preprocessed_{fingerprint}"
            cache_root.mkdir(parents=True, exist_ok=True)

            # Write a human-readable manifest next to the .pt files
            manifest = cache_root / "manifest.txt"
            if not manifest.exists():
                manifest.write_text(
                    f"spacing={self.spacing}\n"
                    f"crop_margin={self.crop_margin}\n"
                    f"spatial_size={self.spatial_size}\n"
                    f"num_samples={self.num_samples}\n"
                    f"fingerprint={fingerprint}\n"
                )

            # Check which samples are already on disk and are not corrupted.
            # A previous interrupted run may leave truncated .pt files that pass
            # an existence check but fail on torch.load.
            pt_paths = [str(cache_root / f"{i:06d}.pt") for i in range(self.num_samples)]

            # Clean up leftover .tmp files from interrupted writes
            for tmp in cache_root.glob("*.tmp"):
                try:
                    tmp.unlink()
                except OSError:
                    pass

            def _is_valid_pt(path):
                """Quick check: file exists and has a plausible size (> 1 KB).

                A truly corrupted file that passes this check will be caught by
                ``_load_cached`` at training time — see the ``except`` branch
                there which deletes and re-raises.
                """
                try:
                    return os.path.getsize(path) > 1024
                except OSError:
                    return False

            missing_indices = [i for i, p in enumerate(pt_paths) if not _is_valid_pt(p)]

            if len(missing_indices) == 0:
                logger.info(
                    f"Loading persistent cache from {cache_root} "
                    f"({self.num_samples} samples, fingerprint={fingerprint})"
                )
                self._cache = [None] * self.num_samples
                self._cache_paths = pt_paths
                self._cache_persistent = True
                # Don't eagerly load — __getitem__ will mmap on first access
                return

            logger.info(
                f"Persistent cache at {cache_root}: "
                f"{self.num_samples - len(missing_indices)}/{self.num_samples} "
                f"already on disk, processing {len(missing_indices)} remaining "
                f"({num_workers} workers)..."
            )

            work_args = [(i, self.items[i], deterministic_transform, pt_paths[i]) for i in missing_indices]

            done = 0
            if num_workers > 0:
                from multiprocessing import Pool

                with Pool(processes=num_workers) as pool:
                    for idx, _ in pool.imap_unordered(
                        MyCustomDataset._transform_and_save,
                        work_args,
                        chunksize=2,
                    ):
                        done += 1
                        if done % 50 == 0 or done == len(missing_indices):
                            print(
                                f"  Processed {done}/{len(missing_indices)}",
                                file=sys.stderr,
                                flush=True,
                            )
            else:
                for idx, _ in map(MyCustomDataset._transform_and_save, work_args):
                    done += 1
                    if done % 50 == 0 or done == len(missing_indices):
                        print(
                            f"  Processed {done}/{len(missing_indices)}",
                            file=sys.stderr,
                            flush=True,
                        )

            logger.info(f"Persistent cache complete: {cache_root}")
            self._cache = [None] * self.num_samples
            self._cache_paths = pt_paths
            self._cache_persistent = True
            return

        # ------------------------------------------------------------------
        # RAM-only cache (original behaviour)
        # ------------------------------------------------------------------
        self._cache_paths = None
        self._cache_persistent = False

        logger.info(f"Caching {self.num_samples} volumes in memory ({num_workers} workers)...")

        work_args = [(i, self.items[i], deterministic_transform) for i in range(self.num_samples)]

        self._cache = [None] * self.num_samples
        mem_bytes = 0
        done = 0

        if num_workers > 0:
            from multiprocessing import Pool

            with Pool(processes=num_workers) as pool:
                for idx, cached in pool.imap_unordered(MyCustomDataset._transform_one, work_args, chunksize=2):
                    self._cache[idx] = cached
                    for v in cached.values():
                        if hasattr(v, "nelement"):
                            mem_bytes += v.nelement() * v.element_size()
                    done += 1
                    if done % 50 == 0 or done == self.num_samples:
                        print(
                            f"  Cached {done}/{self.num_samples} ({mem_bytes / 1e9:.2f} GB)",
                            file=sys.stderr,
                            flush=True,
                        )
        else:
            for idx, cached in map(MyCustomDataset._transform_one, work_args):
                self._cache[idx] = cached
                for v in cached.values():
                    if hasattr(v, "nelement"):
                        mem_bytes += v.nelement() * v.element_size()
                done += 1
                if done % 50 == 0 or done == self.num_samples:
                    print(
                        f"  Cached {done}/{self.num_samples} ({mem_bytes / 1e9:.2f} GB)",
                        file=sys.stderr,
                        flush=True,
                    )

        logger.info(f"Dataset caching complete: {self.num_samples} samples, {mem_bytes / 1e9:.2f} GB")

    # ------------------------------------------------------------------

    def __len__(self):
        return self.num_samples

    def __getview__(self, item):
        """Not used - we override __getitem__ directly."""

    def __get_augmented_view__(self, idx, z, change_list):
        """Not used - T2 is our second view."""

    def sample(self, size, random_state=None):
        """Sample for DCI evaluation - returns empty since we don't have latent factors."""
        return np.array([[]]), []

    def _load_cached(self, idx):
        """Return the cached dict for sample *idx*.

        RAM-only cache: direct lookup.
        Persistent disk cache: load from .pt file each time without retaining
        in ``self._cache`` — the OS page cache keeps hot files in memory, so
        repeated reads are fast without duplicating tensors on the Python heap.
        """
        # RAM-only cache path
        if not self._cache_persistent:
            return self._cache[idx]

        # Persistent disk cache path — load and release, no accumulation
        import torch as _torch

        path = self._cache_paths[idx]
        try:
            return _torch.load(path, map_location="cpu", weights_only=True)
        except RuntimeError as e:
            if "unexpected pos" in str(e) or "invalid load" in str(e).lower():
                # Corrupted file — delete so next run regenerates it
                logger.warning(f"Corrupted cache file (deleting): {path}")
                try:
                    os.remove(path)
                except OSError:
                    pass
                raise RuntimeError(
                    f"Cache file {path} is corrupted and has been deleted. "
                    f"Please restart training to regenerate it."
                ) from e
            raise
        except Exception:
            # Old cache files may contain MONAI MetaTensors with numpy globals
            try:
                cached = _torch.load(path, map_location="cpu", weights_only=False)
                return {
                    k: _torch.as_tensor(v).clone() if hasattr(v, "__torch_function__") else v for k, v in cached.items()
                }
            except Exception:
                logger.warning(f"Corrupted cache file (deleting): {path}")
                try:
                    os.remove(path)
                except OSError:
                    pass
                raise RuntimeError(
                    f"Cache file {path} is corrupted and has been deleted. "
                    f"Please restart training to regenerate it."
                )

    def __getitem__(self, idx):
        """Return dict with T1 and T2 as two views."""
        if self._cache is not None:
            cached = self._load_cached(idx)
            # Clone tensors so augmentations don't mutate the cache
            data_dict = {k: v.clone() if hasattr(v, "clone") else v for k, v in cached.items()}
            if self._aug_transform is not None:
                data_dict = self._aug_transform(data_dict)
        else:
            item = self.items[idx]
            data_dict = {
                "image_t1": item["image"],
                "image_t2": item["z_image"],
                "label": item["label"],
            }
            if "mask_image" in item and "mask_z_image" in item:
                data_dict["mask_t1"] = item["mask_image"]
                data_dict["mask_t2"] = item["mask_z_image"]
            data_dict = self.monai_transform(data_dict)

        img_t1 = data_dict["image_t1"]
        img_t2 = data_dict["image_t2"]
        mask_t1 = data_dict["mask_t1"]
        mask_t2 = data_dict["mask_t2"]

        lbl = data_dict.get("label", -1)
        if hasattr(lbl, "item"):
            lbl = int(lbl.item())
        else:
            try:
                lbl = int(lbl)
            except (TypeError, ValueError):
                lbl = -1

        return {
            "image": [img_t1, img_t2],
            "mask": [mask_t1, mask_t2],
            "z_image": [{}, {}],
            "index": idx,
            "label": lbl,
        }


class SyntheticBrainDataset(MultiviewDataset):
    """Drop-in synthetic baseline mirroring the MyCustomDataset contract.

    Wraps eval.synthetic_dataset.Synthetic3DDisentanglementDataset and emits
    {"image": [v1, v2], "mask": [m1, m2], "z_image": [{}, {}], "index", "label"}
    so the existing training / val loops accept it without changes.

    Most kwargs (spacing, crop_margin, transform, labels_path, masks_dir,
    asymmetric_aug, shared_brain_mask, cache_dir, ...) are accepted for API
    parity but ignored — the synthetic generator owns its own pipeline.
    """

    mean_per_channel = [0.0]
    std_per_channel = [1.0]
    FACTORS = {"image": {0: "view"}}
    DISCRETE_FACTORS = {"image": {}}
    LATENT_SPACES = {"image": {}}

    _SPLIT_OFFSETS = {"train": 0, "val": 1, "test": 2}
    _DEFAULT_SAMPLES = {"train": 1000, "val": 100, "test": 200}

    def __init__(
        self,
        data_dir=None,
        mode="train",
        spatial_size=None,
        cache=False,
        synthetic_mode="pseudo_mri",
        synthetic_seed=42,
        synthetic_num_samples=None,
        synthetic_num_samples_per_mode=None,
        synthetic_n_content=9,
        synthetic_n_style=3,
        synthetic_style_scale=1.0,
        synthetic_content_scale=1.0,
        synthetic_n_deformation_grid=4,
        synthetic_n_fissure_grid=8,
        synthetic_hierarchical_content=False,
        synthetic_normalize="per_sample",
        synthetic_causal=False,
        synthetic_causal_graph="chain",
        synthetic_causal_edge_prob=0.5,
        synthetic_causal_noise_scale=0.4,
        synthetic_causal_nonlinearity="leaky_relu",
        synthetic_clean_content=False,
        synthetic_field_prior="iid",
        synthetic_field_grid=8,
        synthetic_field_kernels="distinct",
        synthetic_field_lengthscales=(1.0, 2.5),
        synthetic_field_tp_dof=8.0,
        synthetic_field_scale=1.0,
        synthetic_lesion_mode="sphere",
        synthetic_lesion_lengthscale=0.4,
        synthetic_lesion_sharpness=10.0,
        synthetic_lesion_threshold=1.0,
        synthetic_wm_softness=0.0,
        synthetic_identifiable_ventricle=False,
        synthetic_content_prior="normal",
        synthetic_content_squash="auto",
        synthetic_content_amp_scale=None,
        synthetic_lesion_radius=0.1,
        synthetic_cortex_parameterization="additive",
        synthetic_center_local_deformations=False,
        synthetic_csf_t1_intensity=0.1,
        **kwargs,
    ):
        super().__init__()
        from eval.synthetic_dataset import Synthetic3DDisentanglementDataset

        self.mode = mode
        self.synthetic_normalize = synthetic_normalize
        # Lazily-estimated global foreground centering/scaling constants for
        # the ``fixed_reference`` normalization mode (see ``_render``).
        self._fixed_mean = None
        self._fixed_scale = None
        # Resolution: cubic. Take min of spatial_size if provided so we don't
        # exceed any axis the user intended; default to 32 (cheap baseline).
        if spatial_size is not None:
            res = int(min(spatial_size))
        else:
            res = 32
        self.res = res

        if synthetic_num_samples is None:
            if synthetic_num_samples_per_mode is not None:
                synthetic_num_samples = synthetic_num_samples_per_mode.get(mode, self._DEFAULT_SAMPLES.get(mode, 100))
            else:
                synthetic_num_samples = self._DEFAULT_SAMPLES.get(mode, 100)

        split_seed = synthetic_seed + self._SPLIT_OFFSETS.get(mode, 0)
        self._inner = Synthetic3DDisentanglementDataset(
            num_samples=synthetic_num_samples,
            res=res,
            seed=split_seed,
            mode=synthetic_mode,
            n_content=synthetic_n_content,
            n_style=synthetic_n_style,
            style_scale=synthetic_style_scale,
            content_scale=synthetic_content_scale,
            n_deformation_grid=synthetic_n_deformation_grid,
            n_fissure_grid=synthetic_n_fissure_grid,
            hierarchical_content=synthetic_hierarchical_content,
            causal=synthetic_causal,
            causal_graph=synthetic_causal_graph,
            causal_edge_prob=synthetic_causal_edge_prob,
            causal_noise_scale=synthetic_causal_noise_scale,
            causal_nonlinearity=synthetic_causal_nonlinearity,
            # UNadjusted seed on purpose: sample noise varies per split (split_seed above),
            # but the causal graph must be shared, or train/val/test are different SCMs.
            scm_seed=synthetic_seed,
            clean_content=synthetic_clean_content,
            field_prior=synthetic_field_prior,
            field_grid=synthetic_field_grid,
            field_kernels=synthetic_field_kernels,
            field_lengthscales=synthetic_field_lengthscales,
            field_tp_dof=synthetic_field_tp_dof,
            field_scale=synthetic_field_scale,
            lesion_mode=synthetic_lesion_mode,
            lesion_lengthscale=synthetic_lesion_lengthscale,
            lesion_sharpness=synthetic_lesion_sharpness,
            lesion_threshold=synthetic_lesion_threshold,
            wm_softness=synthetic_wm_softness,
            identifiable_ventricle=synthetic_identifiable_ventricle,
            content_prior=synthetic_content_prior,
            content_squash=synthetic_content_squash,
            content_amp_scale=synthetic_content_amp_scale,
            lesion_radius=synthetic_lesion_radius,
            cortex_parameterization=synthetic_cortex_parameterization,
            center_local_deformations=synthetic_center_local_deformations,
            csf_t1_intensity=synthetic_csf_t1_intensity,
        )
        if synthetic_normalize in ("per_sample", "shared") and synthetic_n_style > 0:
            import warnings

            warnings.warn(
                f"synthetic_normalize={synthetic_normalize!r} z-scores each volume over its "
                "foreground, which removes exactly the affine intensity map that style applies "
                "(lut = base*gain + bias). Measured: a style swap at fixed anatomy leaves a "
                "residual of 0.08 of the volume's own contrast, vs 0.76 under fixed_reference "
                "(eval/generator_defects.py --tests style). z_style[0] (gain) and z_style[1] "
                "(bias) are effectively erased from the encoder input, so any style-recovery "
                "number from this run is bounded by the normalizer, not by the model. Use "
                "--synthetic-normalize fixed_reference for runs that report style recovery.",
                stacklevel=2,
            )
        self.num_samples = synthetic_num_samples
        self.synthetic_mode = synthetic_mode

        # Optional in-memory cache — synthetic rendering is non-trivial at
        # higher res, and DataLoader workers re-render every epoch otherwise.
        self._cache = [None] * synthetic_num_samples if cache else None

    def __len__(self):
        return self.num_samples

    def __getview__(self, item):
        """Not used — overridden __getitem__."""

    def __get_augmented_view__(self, idx, z, change_list):
        """Not used — second view comes from the synthetic generator."""

    def sample(self, size, random_state=None):
        return np.array([[]]), []

    def _render(self, idx):
        x_v1, x_v2, latents = self._inner[idx]
        if "brain_mask" in latents:
            mask = latents.pop("brain_mask")
            mask_t1 = mask
            mask_t2 = mask.clone()
        else:
            mask_t1 = (x_v1 > 0.05).float()
            mask_t2 = (x_v2 > 0.05).float()

        x_v1, x_v2 = self.normalize_views(x_v1, x_v2, mask_t1, mask_t2)
        return x_v1, x_v2, mask_t1, mask_t2, latents

    def normalize_views(self, x_v1, x_v2, mask_t1, mask_t2):
        """Apply the run's ``--synthetic-normalize`` mode to a rendered view pair.

        Public because an interventional evaluator renders its own pairs and must
        put them through the *identical* normalization the encoder saw in training —
        the mode is not cosmetic (``per_sample`` divides out global gain, so an
        intervention on a global factor is partly cancelled by the normalizer).
        """
        if self.synthetic_normalize == "shared":
            m = mask_t1 > 0
            if m.any():
                vals = x_v1[m]
                mean = vals.mean()
                std = vals.std().clamp_min(1e-6)
                x_v1 = (x_v1 - mean) / std * mask_t1
                x_v2 = (x_v2 - mean) / std * mask_t2
        elif self.synthetic_normalize == "fixed_reference":
            if self._fixed_mean is None:
                self._compute_fixed_reference()
            x_v1 = (x_v1 - self._fixed_mean) / self._fixed_scale * mask_t1
            x_v2 = (x_v2 - self._fixed_mean) / self._fixed_scale * mask_t2
        else:
            x_v1 = self._znorm_nonzero(x_v1, mask_t1)
            x_v2 = self._znorm_nonzero(x_v2, mask_t2)

        return x_v1, x_v2

    def _compute_fixed_reference(self, n_ref=64, scale_quantile=0.99):
        """Estimate global foreground centering/scaling constants.

        Used by the ``fixed_reference`` normalization mode: every sample and
        view is standardized by these dataset-level constants instead of its
        own statistics, so global per-sample intensity factors (style gain and
        bias) are preserved into the encoder input rather than divided out.

        The scale is a *robust* spread — the ``scale_quantile`` quantile of the
        absolute foreground deviation, not the standard deviation — so the bulk
        of the (globally mean-shifted) tissue maps into roughly [-1, 1].  This
        matters because the reconstruction loss clamps predictions to [-1, 1]
        with no gradient beyond: a std-based scale leaves bright tissue at ~+2,
        so the loss would flatten every bright voxel to +1.  Per-sample gain/
        bias survives as relative differences under this shared affine map.
        """
        n = min(n_ref, len(self._inner))
        vals_all = []
        for j in range(n):
            x_v1, x_v2, latents = self._inner[j]
            bm = latents.get("brain_mask", None) if isinstance(latents, dict) else None
            if bm is not None:
                masks = (bm > 0, bm > 0)
            else:
                masks = (x_v1 > 0.05, x_v2 > 0.05)
            for x, mk in ((x_v1, masks[0]), (x_v2, masks[1])):
                vals = x[mk]
                if vals.numel():
                    vals_all.append(vals.flatten().float())
        if not vals_all:
            self._fixed_mean, self._fixed_scale = 0.0, 1.0
            return
        vals_all = torch.cat(vals_all)
        # torch.quantile caps at ~16M elements; a uniform subsample is plenty
        # for a stable quantile and keeps the estimate memory-cheap.
        cap = 4_000_000
        if vals_all.numel() > cap:
            g = torch.Generator().manual_seed(0)
            vals_all = vals_all[torch.randint(vals_all.numel(), (cap,), generator=g)]
        mean = float(vals_all.mean())
        scale = float(torch.quantile((vals_all - mean).abs(), scale_quantile))
        self._fixed_mean = mean
        self._fixed_scale = max(scale, 1e-6)

    @staticmethod
    def _znorm_nonzero(x, mask):
        m = mask > 0
        if m.any():
            vals = x[m]
            mean = vals.mean()
            std = vals.std().clamp_min(1e-6)
            x = (x - mean) / std
            x = x * mask
        return x

    def __getitem__(self, idx):
        if self._cache is not None and self._cache[idx] is not None:
            x_v1, x_v2, mask_t1, mask_t2, latents = self._cache[idx]
        else:
            x_v1, x_v2, mask_t1, mask_t2, latents = self._render(idx)
            if self._cache is not None:
                self._cache[idx] = (x_v1, x_v2, mask_t1, mask_t2, latents)

        return {
            "image": [x_v1, x_v2],
            "mask": [mask_t1, mask_t2],
            "z_image": [{}, {}],
            "index": idx,
            "label": 0,
            # Ground-truth latents — not consumed by training, but available
            # for downstream R²/DCI probes via `data["gt_latents"]`.
            "gt_latents": latents,
        }
