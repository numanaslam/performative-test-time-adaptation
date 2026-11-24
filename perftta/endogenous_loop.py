"""Endogenous performative loop: self-induced collapse on a BALANCED, IID stream, with
NO eps knob and NO exogenous correlation. The ONLY feedback channel is the model's own
confidence-gated REPLAY BUFFER -- the memory-bank / reliable-sample mechanism of deployed
TTA (EATA/CoTTA), made into a closed loop.

Each step: a fresh class-BALANCED iid batch arrives. The model's confident samples
(entropy < tau) are pushed into a persistent buffer (FIFO, capacity C). The model then
adapts on a batch that is an `alpha` fraction drawn from that self-selected buffer and the
rest fresh. Because confident samples are disproportionately the model's currently-favored
classes, the buffer's TRUE class marginal concentrates as the model drifts -- so any class
imbalance here is entirely MODEL-INDUCED on balanced input. The distribution map is
D(theta) = law of the confidence-gated buffer; `alpha` is the endogenous performativity
strength (alpha=0 trains on fresh only = no feedback / control; alpha=1 = full feedback).

Reported: balanced (class-uniform) accuracy, stream accuracy on the fresh batch, and
u_buf = (logK - H(q_buf))/logK -- the theory's pressure u, measured on the buffer's true
class marginal, with no eps and no correlation.
"""
import math
import torch
from .utils import softmax_entropy
from .metrics import accuracy, pred_hist, entropy_of_dist, max_class_share, time_to_collapse


def run_endogenous_loop(adapter, pool, num_classes, steps, batch_size, buffer_cap=2000,
                        alpha=1.0, select_tau=None, seed=0, eval_every=0, eval_bs=256):
    g = torch.Generator().manual_seed(seed)
    eg = torch.Generator().manual_seed(seed + 777)
    uniform = torch.ones(num_classes) / num_classes
    Hmax = math.log(num_classes)
    buf_x, buf_y = [], []                          # confidence-gated memory (true y tracked for measurement)
    n_buf = int(round(alpha * batch_size))
    n_fresh = batch_size - n_buf
    log = {"acc": [], "Hq": [], "Hbuf": [], "share_buf": [], "buf_size": [], "bacc": []}

    for t in range(steps):
        x, y = pool.sample(uniform, batch_size, generator=g)     # BALANCED iid -- no correlation, no eps
        with torch.no_grad():
            logits = adapter.predict(x)
        log["acc"].append(accuracy(logits, y))
        log["Hq"].append(entropy_of_dist(pred_hist(logits.detach().cpu(), num_classes)))

        # endogenous channel: push the model's CONFIDENT samples into the persistent buffer
        ent = softmax_entropy(logits).detach().cpu()
        keep = (ent < select_tau) if select_tau is not None else torch.ones_like(ent, dtype=torch.bool)
        for i in torch.nonzero(keep, as_tuple=True)[0].tolist():
            buf_x.append(x[i:i + 1].detach().clone())
            buf_y.append(int(y[i].item()))
        while len(buf_x) > buffer_cap:                           # FIFO eviction
            buf_x.pop(0); buf_y.pop(0)

        # measure the buffer's TRUE class marginal = the model-induced imbalance on balanced input
        if buf_y:
            bh = torch.bincount(torch.tensor(buf_y), minlength=num_classes).float()
            bh = bh / bh.sum()
        else:
            bh = uniform.clone()
        log["Hbuf"].append(entropy_of_dist(bh))
        log["share_buf"].append(max_class_share(bh))
        log["buf_size"].append(len(buf_x))

        if eval_every and ((t + 1) % eval_every == 0):
            xb, yb = pool.sample(uniform, eval_bs, generator=eg)
            with torch.no_grad():
                log["bacc"].append(accuracy(adapter.predict(xb), yb))

        # train on alpha fraction from the self-selected buffer, rest fresh
        if n_buf > 0 and len(buf_x) >= 1:
            idx = torch.randint(len(buf_x), (n_buf,), generator=g).tolist()
            x_buf = torch.cat([buf_x[i] for i in idx], 0)
            x_train = torch.cat([x_buf, x[:n_fresh]], 0) if n_fresh > 0 else x_buf
        else:
            x_train = x
        adapter.update(x_train)

    n_tail = min(10, len(log["Hbuf"]))
    log["ttc"] = time_to_collapse(log["share_buf"])
    log["final_Hbuf"] = sum(log["Hbuf"][-n_tail:]) / n_tail
    log["final_Hq"] = sum(log["Hq"][-n_tail:]) / n_tail
    log["final_acc"] = sum(log["acc"][-n_tail:]) / n_tail
    log["final_buf_size"] = log["buf_size"][-1] if log["buf_size"] else 0
    log["u_buf"] = max(0.0, (Hmax - log["final_Hbuf"]) / Hmax)   # endogenous performativity pressure
    if log["bacc"]:
        nb = min(3, len(log["bacc"]))
        log["init_bacc"] = log["bacc"][0]
        log["final_bacc"] = sum(log["bacc"][-nb:]) / nb
    else:
        log["init_bacc"] = float("nan")
        log["final_bacc"] = float("nan")
    return log
