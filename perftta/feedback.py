"""The performative distribution map D(theta): how the model's induced class
marginal reshapes the next batch's class prior.

    pi_{t+1} = (1 - eps) * pi_0 + eps * q(theta_t)

eps = 0  -> ordinary (exogenous) TTA; the next distribution ignores the model.
eps > 0  -> the self-reinforcing loop. eps is the performativity strength.
"""
import torch


class PerformativeFeedback:
    def __init__(self, num_classes, eps, base_prior=None):
        self.K = int(num_classes)
        self.eps = float(eps)
        if base_prior is None:
            base_prior = torch.ones(self.K) / self.K
        self.base_prior = base_prior / base_prior.sum()

    def next_prior(self, q_hat):
        q = q_hat / q_hat.sum().clamp_min(1e-12)
        pi = (1.0 - self.eps) * self.base_prior + self.eps * q
        return pi / pi.sum()
