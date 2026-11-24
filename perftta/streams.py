"""Realistic (non-iid) test streams for the NATURAL-feedback experiment.

The main performative loop (`loop.py`) injects feedback through an explicit class-prior
knob `eps`. To show the collapse is not an artifact of that knob, the natural experiment
uses an EXOGENOUS, temporally-correlated stream -- the standard online-TTA deployment
condition (RoTTA/NOTE) -- and lets the *endogenous* feedback come only from the adapter's
own confidence-based sample selection (what EATA/SAR already do). This stream does NOT
depend on the model; it is the realistic data-generating process the loop runs on.
"""
import torch


class CorrelatedStream:
    """Markov class process over a class-indexed pool: with prob `rho` keep the current
    class, else resample from `base_prior`. rho=0 -> iid (balanced batches); rho->1 ->
    long single-class runs (strongly imbalanced batches), as in real correlated streams.
    Exogenous: the sequence is independent of the model."""

    def __init__(self, pool, num_classes, rho=0.99, base_prior=None, seed=0):
        self.pool = pool
        self.K = num_classes
        self.rho = float(rho)
        self.base = (base_prior if base_prior is not None
                     else torch.ones(num_classes) / num_classes)
        self.g = torch.Generator().manual_seed(seed + 31)
        self.cur = int(torch.multinomial(self.base, 1, generator=self.g).item())

    def _next_label(self):
        if torch.rand(1, generator=self.g).item() > self.rho:
            self.cur = int(torch.multinomial(self.base, 1, generator=self.g).item())
        return self.cur

    def next_batch(self, batch_size):
        labels = [self._next_label() for _ in range(batch_size)]
        return self.pool.draw(labels, generator=self.g)
