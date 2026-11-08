"""Standard-error ratio of central finite-difference delta vs the Malliavin weight,
digital call, S0=K=100, T=1, r=5%, q=2%, sigma=20%. Common random numbers for FD."""
import numpy as np

S0 = K = 100.0
T, r, q, s = 1.0, 0.05, 0.02, 0.20


def ratios(n=1_000_000, seed=0, hs=(0.01, 0.001)):
    Z = np.random.default_rng(seed).standard_normal(n)
    disc = np.exp(-r * T)

    def ST(S):
        return S * np.exp((r - q - 0.5 * s * s) * T + s * np.sqrt(T) * Z)

    mal = disc * (ST(S0) > K) * Z / (S0 * s * np.sqrt(T))
    out = {}
    for hr in hs:
        h = hr * S0
        fd = disc * ((ST(S0 + h) > K).astype(float) - (ST(S0 - h) > K)) / (2 * h)
        out[hr] = fd.std() / mal.std()
    return out


if __name__ == "__main__":
    for h, v in ratios().items():
        print(f"FD h={h:.1%}: SE is {v:.1f}x the Malliavin SE")
