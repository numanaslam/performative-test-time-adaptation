"""Collapse / stability / performance metrics for the performative loop."""
import torch


def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def pred_hist(logits, num_classes):
    """Normalized histogram of argmax predictions -> empirical induced marginal q(theta)."""
    h = torch.bincount(logits.argmax(1), minlength=num_classes).float()
    return h / h.sum().clamp_min(1.0)


def entropy_of_dist(p, eps=1e-12):
    p = p.clamp_min(eps)
    return float(-(p * p.log()).sum())


def max_class_share(p):
    return float(p.max())


def time_to_collapse(shares, tau=0.5, window=5):
    """First step where the dominant predicted class holds >= tau for `window`
    consecutive steps. Returns len(shares) if the loop never collapses."""
    run = 0
    for t, s in enumerate(shares):
        run = run + 1 if s >= tau else 0
        if run >= window:
            return t - window + 1
    return len(shares)


def performative_regret(adapter_errs, source_errs):
    """Cumulative online-error gap vs the frozen source on the SAME induced stream.
    Positive => the adapter did worse than staying frozen (a proxy for negative,
    self-induced transfer)."""
    a = torch.tensor(adapter_errs).cumsum(0)
    s = torch.tensor(source_errs).cumsum(0)
    return (a - s).tolist()
