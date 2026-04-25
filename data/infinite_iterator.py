from typing import Iterable

import torch
from torch.utils.data import Sampler


class InfiniteIterator:
    """Infinitely repeat the iterable."""

    def __init__(self, iterable: Iterable):
        self._iterable = iterable
        self.iterator = iter(self._iterable)

    def __iter__(self):
        return self

    def __next__(self):
        for _ in range(2):
            try:
                return next(self.iterator)
            except StopIteration:
                # reset iterator
                del self.iterator
                self.iterator = iter(self._iterable)


class ResumableSampler(Sampler):
    """Random sampler whose state survives a checkpoint/resume.

    Each epoch produces a permutation seeded by ``seed + epoch`` (deterministic
    given the seed), and ``consumed`` indices already yielded in the current
    epoch are skipped on iteration. State_dict captures both, so resuming
    continues the same epoch from the same offset rather than restarting at
    epoch 0 — which is what causes MoCo queues / VQ-codebook EMA / dead-code
    resampling to see a different sample stream than continuous training and
    transiently regress.

    Note: with prefetching workers, ``consumed`` may slightly overshoot the
    number of batches the model actually consumed (by up to
    ``prefetch_factor * num_workers`` batches). Acceptable: a handful of
    samples are skipped per resume.
    """

    def __init__(self, data_source, seed: int = 0):
        self.n = len(data_source)
        self.seed = int(seed)
        self.epoch = 0
        self.consumed = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.n, generator=g).tolist()
        start = self.consumed
        for idx in perm[start:]:
            yield idx
            self.consumed += 1
        self.epoch += 1
        self.consumed = 0

    def __len__(self):
        return max(0, self.n - self.consumed)

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "consumed": self.consumed, "seed": self.seed}

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        self.epoch = int(state.get("epoch", 0))
        self.consumed = int(state.get("consumed", 0))
        self.seed = int(state.get("seed", self.seed))
