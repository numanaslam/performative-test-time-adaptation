"""Adapters: Source (frozen), TENT, EATA-lite (selection), PATTA (ours).

All adapters expose:  predict(x) -> logits (no grad) ;  update(x) -> None
TENT/EATA/PATTA adapt only BN affine params and use batch statistics (the TENT
recipe). PATTA adds the marginal-entropy (class-balance) term that we DERIVE as
the performative-stability correction, with an adaptive weight tied to measured
loop pressure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import softmax_entropy, marginal_entropy


def configure_tent(model):
    """Freeze everything except BN affine params; make BN use batch stats."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.train()
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def bn_params(model):
    ps = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            for p in m.parameters():
                if p.requires_grad:
                    ps.append(p)
    return ps


def _make_opt(params, lr, optim):
    if optim == "adam":
        return torch.optim.Adam(params, lr=lr)
    return torch.optim.SGD(params, lr=lr, momentum=0.0)


class SourceFrozen:
    name = "source"

    def __init__(self, model, **kw):
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def predict(self, x):
        return self.model(x)

    def update(self, x):
        pass


class Tent:
    name = "tent"

    def __init__(self, model, lr=1e-3, optim="sgd", **kw):
        self.model = configure_tent(model)
        self.opt = _make_opt(bn_params(self.model), lr, optim)

    @torch.no_grad()
    def predict(self, x):
        return self.model(x)

    def update(self, x):
        self.opt.zero_grad()
        loss = softmax_entropy(self.model(x)).mean()
        loss.backward()
        self.opt.step()


class EataLite(Tent):
    name = "eata_lite"

    def __init__(self, model, lr=1e-3, num_classes=10, margin_frac=0.4, **kw):
        super().__init__(model, lr=lr, **kw)
        self.e_margin = margin_frac * torch.log(torch.tensor(float(num_classes))).item()

    def update(self, x):
        self.opt.zero_grad()
        ent = softmax_entropy(self.model(x))
        mask = ent < self.e_margin            # keep only confident samples (selection)
        if mask.any():
            ent[mask].mean().backward()
            self.opt.step()


class Patta(Tent):
    name = "patta"

    def __init__(self, model, lr=1e-3, lam0=1.0, num_classes=10, adaptive=True,
                 p=1.0, lam_min=0.0, **kw):
        super().__init__(model, lr=lr, **kw)
        self.lam0 = float(lam0)
        self.adaptive = adaptive
        self.p = float(p)                     # ramp exponent: p<1 engages the correction earlier
        self.lam_min = float(lam_min)         # always-on preventive floor (0 => purely reactive)
        self.Hmax = torch.log(torch.tensor(float(num_classes))).item()

    def _lam(self, Hq_item):
        """Marginal-entropy weight from measured loop pressure u = (Hmax-H(q))/Hmax
        in [0,1] (0 = uniform marginal, 1 = fully collapsed). Defaults p=1, lam_min=0
        reproduce the original reactive linear ramp lam0*u exactly."""
        if not self.adaptive:
            return self.lam0                  # fixed-weight (BalanceFixed)
        u = max(0.0, (self.Hmax - Hq_item) / self.Hmax)
        return max(self.lam_min, self.lam0 * (u ** self.p))

    def update(self, x):
        self.opt.zero_grad()
        logits = self.model(x)
        probs = logits.softmax(1)
        ent = softmax_entropy(logits).mean()
        qbar = probs.mean(0)                  # differentiable estimate of induced marginal
        Hq = marginal_entropy(qbar)
        lam = self._lam(Hq.item())
        loss = ent - lam * Hq                 # reward a spread-out marginal (stability correction)
        loss.backward()
        self.opt.step()


class PseudoLabel(Tent):
    """Hard self-training: minimize cross-entropy to the model's OWN argmax
    predictions. The canonical collapse-prone TTA objective -- used to probe
    whether performative feedback drives collapse once the mechanism is not
    benign."""
    name = "plst"

    def update(self, x):
        self.opt.zero_grad()
        logits = self.model(x)
        pl = logits.argmax(1).detach()
        F.cross_entropy(logits, pl).backward()
        self.opt.step()


class Sar(Tent):
    """SAR-lite (Niu et al. 2023): entropy-margin sample selection + sharpness-
    aware minimization (SAM) on BN-affine params. The model-reset recovery is
    omitted. A collapse-mitigation BASELINE to test PATTA against."""
    name = "sar"

    def __init__(self, model, lr=1e-3, num_classes=10, margin_frac=0.4, rho=0.05, **kw):
        super().__init__(model, lr=lr, **kw)
        self.e_margin = margin_frac * torch.log(torch.tensor(float(num_classes))).item()
        self.rho = rho
        self.params = [p for p in bn_params(self.model)]

    def update(self, x):
        # first (ascent) step on reliable (low-entropy) samples
        self.opt.zero_grad()
        ent = softmax_entropy(self.model(x))
        mask = ent < self.e_margin
        if not mask.any():
            return
        ent[mask].mean().backward()
        with torch.no_grad():
            gnorm = torch.norm(torch.stack(
                [p.grad.detach().norm() for p in self.params if p.grad is not None]))
            scale = self.rho / (gnorm + 1e-12)
            perturb = []
            for p in self.params:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)                      # climb to the local worst case
                perturb.append((p, e_w))
        # second (descent) step at the perturbed point (sharpness-aware)
        self.opt.zero_grad()
        ent2 = softmax_entropy(self.model(x))
        mask2 = ent2 < self.e_margin
        if mask2.any():
            ent2[mask2].mean().backward()
        with torch.no_grad():
            for p, e_w in perturb:
                p.sub_(e_w)                      # restore original params
        self.opt.step()


class BalanceFixed(Patta):
    """Fixed-weight marginal-entropy (class-balance) regularizer: the HEURISTIC
    form of PATTA's correction, with a constant lambda instead of the
    stability-derived adaptive schedule. Isolates what the adaptive weight adds
    (and, unlike PATTA, it is NOT inert in the benign regime)."""
    name = "balance"

    def __init__(self, model, lr=1e-3, lam0=1.0, num_classes=10, **kw):
        kw.pop("adaptive", None)
        super().__init__(model, lr=lr, lam0=lam0, num_classes=num_classes,
                         adaptive=False, **kw)


class PattaSteep(Patta):
    """PATTA with a CONCAVE ramp (p<1): the correction engages at the first sign of
    marginal concentration rather than after it, catching collapse before it deepens.
    Still exactly inert when the marginal is uniform (lam_min=0), so PATTA's benign-
    regime safety property is preserved. Isolates the 'engage earlier' ingredient."""
    name = "patta_steep"

    def __init__(self, model, p=0.5, **kw):
        kw.pop("adaptive", None)
        super().__init__(model, adaptive=True, p=p, lam_min=0.0, **kw)


class PattaFloor(Patta):
    """PATTA with an always-on preventive floor (lam_min>0): a weak class-balance
    pressure applies from step 0 (a mild fixed BalanceFixed), with the adaptive term
    adding strength under pressure. NOT inert in the benign regime -- that floor tax
    is measured directly. Isolates the 'preventive-from-step-0' ingredient."""
    name = "patta_floor"

    def __init__(self, model, lam_min=0.1, **kw):
        kw.pop("adaptive", None)
        super().__init__(model, adaptive=True, lam_min=lam_min, **kw)


class PattaSev(Patta):
    """Severity-aware PATTA (the proposal): concave ramp + preventive floor, at the
    SAME max strength as BalanceFixed (lam0=1.0) -- only the schedule shape differs.
    Preventive early, full strength under deep collapse. Targets the CIFAR-100 near-
    total-collapse regime where plain PATTA under-corrects, while aiming to keep
    PATTA's CIFAR-10 advantage and near-inertness in the benign regime."""
    name = "patta_sev"

    def __init__(self, model, p=0.5, lam_min=0.1, **kw):
        kw.pop("adaptive", None)
        super().__init__(model, adaptive=True, p=p, lam_min=lam_min, **kw)


class Scalar(Tent):
    """SCALAR-lite (Amar & Lindenbaum, NeurIPS 2026): control the pseudo-label CLASS
    MARGINAL with Sinkhorn normalization, then self-train to the class-balanced soft
    targets. The closest PRIOR method to PATTA's correction (explicit marginal control),
    included as the strongest anti-collapse baseline. PATTA differs by DERIVING the
    marginal term from performative stability and ADAPTING its weight, rather than
    projecting to a fixed uniform marginal every step."""
    name = "scalar"

    def __init__(self, model, lr=1e-3, num_classes=10, n_sink=3, tau=0.05, **kw):
        super().__init__(model, lr=lr, **kw)
        self.n_sink = int(n_sink)
        self.tau = float(tau)

    @staticmethod
    def _sinkhorn(logits, n_iter, tau):
        Q = torch.exp(logits / tau).t()                    # (K, B)
        Q = Q / (Q.sum() + 1e-12)
        K, B = Q.shape
        for _ in range(n_iter):
            Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-12) / K   # each class row -> 1/K
            Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-12) / B   # each sample col -> 1/B
        return (Q * B).t()                                 # (B, K), rows sum to 1

    def update(self, x):
        self.opt.zero_grad()
        logits = self.model(x)
        with torch.no_grad():
            targets = self._sinkhorn(logits.detach(), self.n_sink, self.tau)
        loss = -(targets * logits.log_softmax(1)).sum(1).mean()
        loss.backward()
        self.opt.step()


class SarReset(Sar):
    """SAR *with* the model-reset recovery (Niu et al. 2023): when the predicted-class
    marginal concentrates past a threshold (an EMA of its max class share), reset the
    BN-affine params to source. This is the recovery mechanism our `sar` (lite) omitted,
    and the reviewer-requested fuller SAR for the anti-collapse comparison."""
    name = "sar_reset"

    def __init__(self, model, lr=1e-3, optim="sgd", num_classes=10,
                 reset_share=0.5, ema=0.9, **kw):
        super().__init__(model, lr=lr, optim=optim, num_classes=num_classes, **kw)
        self.src = [p.detach().clone() for p in self.params]
        self.K = int(num_classes); self.lr = float(lr); self.optim_name = optim
        self.reset_share = float(reset_share); self.ema_beta = float(ema)
        self.share_ema = 1.0 / self.K; self.n_resets = 0

    def update(self, x):
        super().update(x)                                  # SAR selection + SAM step
        with torch.no_grad():
            share = self.model(x).softmax(1).mean(0).max().item()
        self.share_ema = self.ema_beta * self.share_ema + (1 - self.ema_beta) * share
        if self.share_ema > self.reset_share:              # collapse detected -> reset to source
            with torch.no_grad():
                for p, s in zip(self.params, self.src):
                    p.data.copy_(s)
            self.opt = _make_opt(self.params, self.lr, self.optim_name)
            self.share_ema = 1.0 / self.K; self.n_resets += 1


class BalanceLo(BalanceFixed):
    """Tuned fixed marginal-entropy baseline, weak weight (lam=0.5)."""
    name = "balance_lo"

    def __init__(self, model, **kw):
        kw.pop("lam0", None)
        super().__init__(model, lam0=0.5, **kw)


class BalanceHi(BalanceFixed):
    """Tuned fixed marginal-entropy baseline, strong weight (lam=2.0)."""
    name = "balance_hi"

    def __init__(self, model, **kw):
        kw.pop("lam0", None)
        super().__init__(model, lam0=2.0, **kw)


ADAPTERS = {c.name: c for c in
            [SourceFrozen, Tent, EataLite, Sar, SarReset, BalanceFixed, BalanceLo, BalanceHi,
             Scalar, Patta, PattaSteep, PattaFloor, PattaSev, PseudoLabel]}


def build_adapter(name, model, **kw):
    return ADAPTERS[name](model, **kw)
