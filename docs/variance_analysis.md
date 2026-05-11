# Variance Analysis: When to Use Malliavin vs Alternatives

## Summary

| Setting | Best method | Why |
|---------|-------------|-----|
| European call delta (GBM) | FD with CRN | FD variance ≈ 0.5× Malliavin variance |
| Digital call delta | Malliavin | FD variance → ∞ as h → 0 |
| Barrier option delta (near B) | Malliavin | FD variance explodes near the barrier |
| Asian call delta | Malliavin or pathwise | Single simulation; pathwise zero-variance for smooth payoffs |
| Lookback delta | Malliavin (BEL) | Pathwise requires max derivative (discontinuous) |
| Gamma, all payoffs | Malliavin | FD requires 2 extra sims; Malliavin weight is direct |
| Vanna, Volga | Malliavin | FD cross-difference requires 4 sims; Malliavin is 1 sim |

---

## 1. Why FD Fails for Discontinuous Payoffs

### 1.1 The Indicator Function Problem

For a digital call f(S_T) = 1_{S_T > K}, the finite-difference delta estimator is:

```
δ̂_FD = e^{-rT} · [f(S_T^{S₀+h}) − f(S_T^{S₀−h})] / (2h)
```

The payoff difference is:

```
f(S_T^+) − f(S_T^-) = 1_{S_T^+ > K} − 1_{S_T^- > K}
```

This equals ±1 on paths where one simulation crosses K and the other does not, and 0 everywhere else.  
The probability of crossing is ≈ φ(d₂)·2h/(σS₀√T) for small h (where φ is the normal pdf), so:

```
Var[f(S_T^+) − f(S_T^-)] ≈ φ(d₂)·2h/(σS₀√T) · [1 − φ(d₂)·2h/(σS₀√T)] ≈ C·h
```

Dividing by (2h)²:

```
Var[δ̂_FD] ≈ C·h / (4h²) = C/(4h) → ∞ as h → 0
```

**The FD variance of digital delta diverges as h → 0.** Increasing paths does not help: for n paths, the MSE = Var/(n) + bias² with bias ∝ h², so the optimal h* ∝ n^{-1/3} gives MSE ∝ n^{-2/3} — slower than the Monte Carlo rate n^{-1}.

### 1.2 Variance Bound for the Indicator Function

Let X ~ N(μ, σ²) and f(x) = 1_{x > K}. Then:

```
E[1_{X>K}·(x − μ)/σ²] = φ((K−μ)/σ)/σ   (by the Gaussian IBP)
```

The weight (X − μ)/σ² is square-integrable, so E[(1_{X>K}·(X−μ)/σ²)²] < ∞ — **finite variance**.

This is the prototypical Malliavin integration by parts: the indicator function is never differentiated.

---

## 2. Why Malliavin Succeeds for Discontinuous Payoffs

Under GBM, the Malliavin delta estimator for any payoff f is:

```
δ̂_Mall = e^{-rT} · f(S_T) · W_T/(σS₀T)
```

For the digital call f = 1_{S_T > K}:

```
Var[δ̂_Mall] = disc² · E[(1_{S_T>K} · W_T/(σS₀T))²]
             = disc² · E[W_T²/(σ²S₀²T²) | S_T > K] · P(S_T > K)
             ≤ disc² · E[W_T²]/(σ²S₀²T²)
             = disc² · T/(σ²S₀²T²)
             = disc² / (σ²S₀²T)
```

This is **finite and independent of h** (there is no h in the Malliavin estimator). The variance is of the same order as for a European call, and scales as 1/n_paths.

---

## 3. Variance Comparison: Smooth Payoffs

For a **European call** f(S_T) = max(S_T − K, 0), the FD delta (central difference with CRN) has variance:

```
Var[δ̂_FD] ≈ disc² · σ²S₀²/(4h²T) · Var[ΔW_T] · |f''(S_T)|²  ... (schematic)
```

More precisely, under CRN (same Brownian path ±h):

```
Var[(f(S_T^+) − f(S_T^-))/(2h)] ≈ Var[f'(S_T)] · σ²S₀²/(4T)   (for small h)
```

The FD variance converges to **the pathwise/IPA variance** Var[f'(S_T)·∂S_T/∂S₀] as h → 0.

For GBM and European call:
- **Pathwise/IPA variance** ≈ (S₀·e^{(r−q)T}·N(d₁))² · T/n_paths (zero-variance limit with exact AD)
- **Malliavin variance** = Var[1_{S_T>K}·W_T/(σS₀T)] — larger because it evaluates at zero slope (the payoff kink)

**Conclusion:** For smooth payoffs, FD (with small enough h) achieves lower variance than Malliavin because Malliavin uses the score function which has non-zero variance even where f'(S_T) = 0 (out-of-the-money paths still contribute W_T/(σS₀T) to the estimator).

For vanilla options with n_steps = 1, empirically:
- Malliavin SE / FD SE ≈ 1.5–2.5 (FD wins by 2–6× in variance)
- But Malliavin computes ALL Greeks in one simulation; FD needs 2× more simulations per Greek

---

## 4. Variance Bounds on Malliavin Estimators

**Theorem (Fournié et al., 1999):** For any payoff f with E[f(S_T)²] < ∞:

```
Var[f(S_T) · π_θ] ≤ E[f(S_T)²] · E[π_θ²]
```

by Cauchy–Schwarz. The weight second moment E[π_θ²] is model-dependent:

| Model | E[π_Δ²] (delta weight) | Notes |
|-------|----------------------|-------|
| GBM | 1/(σ²S₀²T) | Minimal — GBM has the simplest weight |
| Local vol (CEV β=0.7) | ≈ 2–3/(σ²S₀²T) | First variation has more variance |
| Heston | O(1/(V₀S₀²T)) × (correction) | 1/√V factor dominates near zero |

For the GBM case, the optimal localisation (Fournié et al., 2001) achieves:

```
Var[π_Δ^{optimal}] = (1/T) ∫₀ᵀ E[(Y_s/σ(S_s))²] ds ≥ Var[π_Δ^{uniform}]
```

with equality when Y_s/σ(S_s) is constant (GBM — the optimal localisation is already achieved by the uniform kernel u = 1/T).

For non-GBM models, the optimal localisation **concentrates weight on time intervals where Y_s/σ(S_s) has smallest variance** — typically the early time steps when the path uncertainty is smallest.

---

## 5. Path-Dependent Payoffs: Malliavin Wins on Efficiency

For path-dependent payoffs (Asian, barrier, lookback) under GBM:

**FD delta:** requires 2 full path simulations (S₀ + h and S₀ − h), each with n_steps steps.  
**Malliavin:** requires 1 simulation; the weight is just ΔW₁/(σS₀Δt₁).

For Asians and barriers, both give the same SE per path (both use S₀ → S_T information), but Malliavin computes ALL Greeks (delta, gamma, vega) from the same simulation — 3× more efficient.

**Pathwise/IPA delta** for Asian call:
- IPA: δ = disc · E[f'(Ā) · Ā/S₀] where Ā = (1/n)Σ S_{tᵢ} (arithmetic average)
- For European payoff on Ā: f'(Ā) = 1_{Ā>K} — still discontinuous as function of S₀!
- IPA variance is similar to Malliavin for Asians; both are O(1/n)

**Barrier options near the barrier:**  
As S₀ → B (barrier level from above), the number of paths that survive changes rapidly.  
FD variance explodes because bumping S₀ by h changes the fraction of survived paths by O(h/distance_to_barrier):

```
Var[δ̂_FD] ∝ 1 / (distance_to_barrier · h)  → ∞ as S₀ → B
```

Malliavin uses the weight W_T/(σS₀T) — the barrier indicator 1_{survival} is part of f, not differentiated. The estimator variance remains O(1/(σ²S₀²T)) uniformly in S₀, including near B.

---

## 6. Higher-Order Greeks: FD Instability

### 6.1 FD Gamma

Central-difference gamma requires **three price evaluations** (S₀−h, S₀, S₀+h):

```
Γ̂_FD = [V(S₀+h) − 2V(S₀) + V(S₀−h)] / h²
```

The statistical error scales as SE(V̂)/h², and the optimal h* ∝ n^{-1/4} gives MSE ∝ n^{-1/2} — much worse than n^{-1}.

Malliavin gamma uses the closed-form weight π_Γ on **the same simulation** as the delta. Cost: 1 simulation for delta AND gamma AND vega combined.

### 6.2 Digital Call Gamma: The Derivative of a Delta Function

For f(S_T) = 1_{S_T > K}, the gamma ∂²V/∂S₀² involves:

```
∂²/∂S₀² E[1_{S_T>K}] = E[δ'(S_T − K) · (∂S_T/∂S₀)²]
```

where δ'(x) is the **derivative of the Dirac delta** — an object of infinite variance in MC.  
FD gamma for a digital has variance O(1/h³) → ∞ as h → 0. No bump size is acceptable.

Malliavin: E[1_{S_T>K} · π_Γ]. The weight π_Γ is the same closed-form expression regardless of the payoff singularity. Variance is finite (O(E[π_Γ²]) × Cauchy–Schwarz).

### 6.3 Cross-Greeks (Vanna, Volga)

FD cross-difference for vanna requires **4 simulations**:

```
Vanna ≈ [V(S+h_S, σ+h_σ) − V(S+h_S, σ−h_σ) − V(S-h_S, σ+h_σ) + V(S-h_S, σ−h_σ)] / (4 h_S h_σ)
```

Cost: 4× the single simulation cost; variance ∝ 1/(h_S² · h_σ² · n).

Malliavin vanna: computed from the **same simulation** as delta and vega, using the product weight π_vanna = π_Δ · π_v + cross-term. Cost: 0 extra simulations.

---

## 7. Practical Recommendations

```
if payoff is discontinuous (digital, barrier, lookback):
    → use Malliavin (FD variance diverges)
    
elif Greek is second-order (gamma, vanna, volga):
    → use Malliavin (FD instability; Malliavin is same cost as first-order)
    
elif payoff is path-dependent (Asian, lookback, barrier):
    → use Malliavin or pathwise IPA
    → Malliavin: one simulation, all Greeks
    → Pathwise IPA: lower variance for smooth payoffs, more complex implementation
    
elif payoff is smooth and terminal (European call/put):
    if only delta needed:
        → FD with CRN and small h (lower variance than Malliavin for smooth payoffs)
    if multiple Greeks needed:
        → Malliavin (one simulation vs n_Greeks × FD simulations)
    
elif model is non-GBM (Heston, local vol):
    → Malliavin BEL (score function not available; FD still works but expensive)
    → For Heston: expect high weight variance from 1/√V factor
```

---

## 8. Variance Reduction for Malliavin Weights

### 8.1 Antithetic Variates

For Malliavin delta under GBM, the estimator samples_i = f(S_T^i) · W_T^i/(σS₀T).

The antithetic version pairs each path with its reflection (−ΔW):

```
samples_anti = 0.5 · [f(S_T^+) · π^+ + f(S_T^-) · π^-]
```

For the delta weight π = W_T/(σS₀T), π^- = −π^+.  
For a European call where f is increasing and π is odd, f·π is approximately even, so:

```
Cov(f^+·π^+, f^-·π^-) = Cov(f^+·π^+, f^-·(−π^+)) < 0
```

This gives a large variance reduction: VRR = Var_plain/Var_anti ≥ 10 for digital delta.

### 8.2 Control Variates

**Geometric Asian control:** For arithmetic Asian calls, the geometric Asian price has a closed-form formula (modified BS). The arithmetic–geometric correlation is ≥ 0.99 for typical parameters → VRR ≥ 5× for price estimation.

**Delta-hedged portfolio:** z = f(S_T) − Δ_BS · S_T is a martingale under the risk-neutral measure with E[z] ≈ 0 (for at-the-money). Using z as a control variate for f(S_T) exploits the hedging correlation.

### 8.3 Localization

**Truncation:** clip the weight at the q-quantile of |π|:

```
π_R = sign(π) · min(|π|, R)    with R = quantile_{0.999}(|π|)
```

This introduces bias O(exp(−R²/2)) but reduces variance significantly. For the digital call, the weight W_T/(σS₀T) has fat-tailed paths when S_T is near K (high W_T magnitude).

**Interval localization:** use a subset A ⊂ [0,T] of time steps in the BEL integral:

```
π_A = (1/(|A|·Δt)) Σ_{i∈A} ΔWᵢ/(σS₀)
```

This is an unbiased estimator (IBP holds for any subset). Concentrating on the first few steps (smallest Y variance) reduces weight variance for barrier options near the barrier.

---

## 9. References

1. **Fournié et al. (1999)** — Variance bounds and optimality conditions for Malliavin estimators
2. **Fournié et al. (2001)** — Variance reduction: antithetics, control variates, localisation
3. **Gobet & Kohatsu-Higa (2003)** — Localisation for barrier options
4. **Glasserman (2004)** — Chapter 7: bias-variance tradeoff for FD estimators; Chapter 4: variance reduction
5. **L'Ecuyer & Lemieux (2000)** — Common random numbers and antithetics for FD estimators
