"""The performative online loop:  theta_{t+1} = A(theta_t; D(theta_t)).

Each step is strictly causal: the model PREDICTS the arriving batch before it
updates on it, and the prediction it just made induces the NEXT batch's prior.

Optionally tracks BALANCED accuracy: every `eval_every` steps the current model
is evaluated on a fresh class-UNIFORM batch. Mean accuracy on the induced
(imbalanced) stream can stay high while the model abandons suppressed classes;
balanced accuracy exposes that harm.
"""
import torch
from .feedback import PerformativeFeedback
from .metrics import (accuracy, pred_hist, entropy_of_dist, max_class_share,
                      time_to_collapse)


def run_performative_loop(adapter, pool, num_classes, eps, steps, batch_size,
                          base_prior=None, seed=0, eval_every=0, eval_bs=256, eval_chunk=0):
    g = torch.Generator().manual_seed(seed)
    eg = torch.Generator().manual_seed(seed + 10007)
    fb = PerformativeFeedback(num_classes, eps, base_prior)
    prior = fb.base_prior.clone()
    uniform = torch.ones(num_classes) / num_classes
    log = {"acc": [], "err": [], "Hq": [], "share": [], "prior_entropy": [], "bacc": []}

    for t in range(steps):
        x, y = pool.sample(prior, batch_size, generator=g)
        with torch.no_grad():
            logits = adapter.predict(x)
        acc = accuracy(logits, y)
        log["acc"].append(acc)
        log["err"].append(1.0 - acc)
        hist = pred_hist(logits.detach().cpu(), num_classes)   # induced marginal q(theta_t)
        log["Hq"].append(entropy_of_dist(hist))
        log["share"].append(max_class_share(hist))
        log["prior_entropy"].append(entropy_of_dist(prior))

        if eval_every and ((t + 1) % eval_every == 0):          # balanced (class-uniform) probe
            xb, yb = pool.sample(uniform, eval_bs, generator=eg)
            with torch.no_grad():
                if eval_chunk and xb.shape[0] > eval_chunk:      # chunk the forward (big eval, e.g. ImageNet-50)
                    logits_b = torch.cat([adapter.predict(xb[i:i + eval_chunk])
                                          for i in range(0, xb.shape[0], eval_chunk)], 0)
                else:
                    logits_b = adapter.predict(xb)
                log["bacc"].append(accuracy(logits_b, yb))

        adapter.update(x)                                       # commit the adaptation step
        prior = fb.next_prior(hist)                             # performative induction

    n_tail = min(10, len(log["Hq"]))
    log["ttc"] = time_to_collapse(log["share"])
    log["final_Hq"] = sum(log["Hq"][-n_tail:]) / n_tail
    log["mean_acc"] = sum(log["acc"]) / len(log["acc"])
    log["final_acc"] = sum(log["acc"][-n_tail:]) / n_tail
    if log["bacc"]:
        nb = min(3, len(log["bacc"]))
        log["init_bacc"] = log["bacc"][0]
        log["final_bacc"] = sum(log["bacc"][-nb:]) / nb
    else:
        log["init_bacc"] = float("nan")
        log["final_bacc"] = float("nan")
    return log
