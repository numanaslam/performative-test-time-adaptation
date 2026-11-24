"""Numerical verification of Appendix B's tractable model (CPU, numpy only -- no GPU/data).
Reproduces the closed-form constants and the EXACT threshold eps*, and checks the
bifurcation's super-/sub-critical character by continuation of the coupled (theta, pi) map.

Finding (confirms the v0.6 reviewer): the pitchfork is SUBCRITICAL -- nonzero equilibria
exist BELOW eps* and are UNSTABLE; above eps* the balanced state is unstable and the system
collapses discontinuously to the far degenerate attractor. The exact eps* crossing is
correct; the "stable supercritical branches" claim of an earlier draft is NOT.

    python experiments/verify_tractable.py
"""
import numpy as np

Z = np.linspace(-14, 14, 56001)


def model(mu=1.5, sigma=1.0, a=2.0, eta=0.1):
    def Nc(c):
        return np.exp(-0.5 * ((Z - c * mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    Np, Nm = Nc(+1), Nc(-1)
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))
    F = lambda c, t: np.trapz(sig(a * (Z - t)) * (Np if c > 0 else Nm), Z)
    G = lambda c, t: a * a * np.trapz((Z - t) * sig(a * (Z - t)) * (1 - sig(a * (Z - t))) * (Np if c > 0 else Nm), Z)
    h = 1e-4
    S0 = np.trapz(sig(a * Z) * (1 - sig(a * Z)) * Np, Z)
    g1 = G(+1, 0.0)
    d0 = F(+1, 0.0) - F(-1, 0.0)
    rho = (G(+1, h) - G(+1, -h)) / (2 * h)
    estar = rho / (2 * a * S0 * g1 + rho * d0)

    def pistar(t, e):
        Fp, Fm = F(+1, t), F(-1, t)
        return ((1 - e) * 0.5 + e * Fm) / (1 - e * (Fp - Fm))

    def feq(t, e):
        pi = pistar(t, e); return pi * G(+1, t) + (1 - pi) * G(-1, t)

    def eig(t, e):
        pi = pistar(t, e)
        dtg = pi * (G(+1, t + h) - G(+1, t - h)) / (2 * h) + (1 - pi) * (G(-1, t + h) - G(-1, t - h)) / (2 * h)
        dpg = G(+1, t) - G(-1, t)
        dtq = pi * (F(+1, t + h) - F(+1, t - h)) / (2 * h) + (1 - pi) * (F(-1, t + h) - F(-1, t - h)) / (2 * h)
        dpq = F(+1, t) - F(-1, t)
        J = np.array([[1 - eta * dtg, -eta * dpg], [e * dtq, e * dpq]])
        return float(np.max(np.abs(np.linalg.eigvals(J))))

    return dict(S0=S0, g1=g1, d0=d0, rho=rho, estar=estar, feq=feq, eig=eig)


def branches(m, eps):
    ts = np.linspace(0.01, 2.0, 1000)
    fv = np.array([m["feq"](t, eps) for t in ts])
    out = []
    for i in range(1, len(ts)):
        if fv[i - 1] * fv[i] < 0:
            t = ts[i - 1] - fv[i - 1] * (ts[i] - ts[i - 1]) / (fv[i] - fv[i - 1])
            out.append((t, m["eig"](t, eps)))
    return out


if __name__ == "__main__":
    for (mu, sg, a) in [(1.5, 1, 2), (3, 1, 1), (2, 1, 4)]:
        m = model(mu, sg, a)
        print(f"\n(mu,sigma,a)=({mu},{sg},{a}):  S0={m['S0']:.5f} g1={m['g1']:.5f} "
              f"delta0={m['d0']:.5f} rho*={m['rho']:.5f}  eps*={m['estar']:.5f}")
        for e in [0.96 * m["estar"], m["estar"], 1.1 * m["estar"]]:
            br = branches(m, e)
            tag = "below" if e < m["estar"] else ("at" if abs(e - m["estar"]) < 1e-6 else "above")
            desc = ", ".join(f"theta={t:.3f}(|eig|={g:.4f})" for t, g in br) or "no nonzero branch"
            print(f"   eps={e:.4f} ({tag} eps*):  theta0 |eig|={m['eig'](1e-4, e):.4f};  nonzero: {desc}")
    print("\nVERDICT: nonzero branches sit BELOW eps* with |eig|>1 => SUBCRITICAL pitchfork "
          "(exact eps* crossing holds; stable supercritical branches do NOT).")
