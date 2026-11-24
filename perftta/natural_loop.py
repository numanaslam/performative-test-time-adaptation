"""Natural-feedback loop: online TTA on an EXOGENOUS correlated stream (streams.py),
with NO injected performativity knob. The only feedback channel is the adapter's own
confidence-based sample selection -- the endogenous, naturally-occurring performative
channel that reliable-sample TTA (EATA/SAR) already uses in deployment.

We measure the class marginal of the CONFIDENCE-SELECTED samples, q_sel(theta_t): the
distribution the model actually adapts to. Its concentration is the *natural* induced
marginal, and `u_sel = (log K - H(q_sel))/log K` is the theory's performativity pressure
`u` measured without any eps -- directly comparable to the eps-driven pressure in T2.
Balanced (class-uniform) accuracy vs stream accuracy reproduces the hidden-metric test.
"""
import math
import torch
from .utils import softmax_entropy
from .metrics import accuracy, pred_hist, entropy_of_dist, max_class_share, time_to_collapse


def run_natural_loop(adapter, stream, num_classes, steps, batch_size, select_tau=None,
                     seed=0, eval_every=0, eval_bs=256):
    pool = stream.pool
    eg = torch.Generator().manual_seed(seed + 777)
    uniform = torch.ones(num_classes) / num_classes
    Hmax = math.log(num_classes)
    log = {"acc": [], "Hq": [], "Hqsel": [], "share_sel": [], "sel_frac": [], "bacc": []}

    for t in range(steps):
        x, y = stream.next_batch(batch_size)
        with torch.no_grad():
            logits = adapter.predict(x)
        log["acc"].append(accuracy(logits, y))
        cl = logits.detach().cpu()
        log["Hq"].append(entropy_of_dist(pred_hist(cl, num_classes)))   # marginal over all preds

        # the NATURAL induced marginal: class dist of the confidence-selected subset
        ent = softmax_entropy(logits).detach().cpu()
        mask = (ent < select_tau) if select_tau is not None else torch.ones_like(ent, dtype=torch.bool)
        log["sel_frac"].append(float(mask.float().mean()))
        qsel = pred_hist(cl[mask], num_classes) if mask.any() else uniform.clone()
        log["Hqsel"].append(entropy_of_dist(qsel))
        log["share_sel"].append(max_class_share(qsel))

        if eval_every and ((t + 1) % eval_every == 0):                 # balanced probe
            xb, yb = pool.sample(uniform, eval_bs, generator=eg)
            with torch.no_grad():
                log["bacc"].append(accuracy(adapter.predict(xb), yb))

        adapter.update(x)                                              # exogenous stream: no eps feedback

    n_tail = min(10, len(log["Hqsel"]))
    log["ttc"] = time_to_collapse(log["share_sel"])
    log["final_Hqsel"] = sum(log["Hqsel"][-n_tail:]) / n_tail
    log["final_Hq"] = sum(log["Hq"][-n_tail:]) / n_tail
    log["final_acc"] = sum(log["acc"][-n_tail:]) / n_tail
    log["final_sel_frac"] = sum(log["sel_frac"][-n_tail:]) / n_tail
    log["u_sel"] = max(0.0, (Hmax - log["final_Hqsel"]) / Hmax)        # natural performativity pressure
    if log["bacc"]:
        nb = min(3, len(log["bacc"]))
        log["init_bacc"] = log["bacc"][0]
        log["final_bacc"] = sum(log["bacc"][-nb:]) / nb
    else:
        log["init_bacc"] = float("nan")
        log["final_bacc"] = float("nan")
    return log
