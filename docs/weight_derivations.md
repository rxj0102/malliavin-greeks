# Weight Derivations

Detailed derivations of every Malliavin weight implemented in `mgreeks/`.  
All results verified numerically against Black–Scholes analytics.

## Notation

Throughout this document:

| Symbol | Meaning |
|--------|---------|
| S₀ | Initial spot price |
| K | Strike |
| T | Maturity |
| r | Risk-free rate |
| q | Continuous dividend yield |
| σ | Volatility (GBM) |
| μ | Log-drift: r − q − σ²/2 |
| W_T | Terminal Brownian motion = Σ ΔWᵢ |
| Z | Standardised BM: W_T/√T ~ N(0,1) |
| p(·) | Log-normal transition density |
| disc | Discount factor: e^{-rT} |
| f | Payoff function (possibly discontinuous) |
| π_θ | Malliavin weight for Greek ∂V/∂θ |

**Strategy:** express the terminal Brownian W_T as a function of the observable S_T and parameter θ, then differentiate the log-density ∂/∂θ log p(S_T | S₀; θ) holding S_T fixed ("score function" approach). For GBM this is equivalent to the BEL formula because the BEL integrand is path-independent.

---

## 1. GBM Setup

Under GBM (risk-neutral measure):

```
S_T = S₀ exp(μT + σW_T),   W_T ~ N(0,T)
```

Inverting for W_T (holding S_T fixed as a function of the parameter):

```
W_T(θ) = [log(S_T/S₀) − μ(θ)T] / σ(θ)
```

The log-density (up to constants in S_T):

```
log p(S_T | S₀; θ) = −W_T²/(2T) − log σ − ½ log T + const
```

The IBP formula: Greek = disc · E[f(S_T) · π_θ] where

```
π_θ = ∂/∂θ log p = −(W_T/T) · ∂W_T/∂θ − (∂σ/∂θ)/σ  + (discount correction)
```

---

## 2. Delta (∂V/∂S₀)

**Derivation:**

Holding S_T fixed: W_T(S₀) = (log(S_T/S₀) − μT) / σ

```
∂W_T/∂S₀ = −1/(σS₀)   (∂/∂S₀ of log(S_T/S₀) = −1/S₀)
```

Score (∂/∂S₀ log p, holding S_T fixed):

```
∂/∂S₀ log p = −(W_T/T) · (−1/(σS₀)) = W_T/(σS₀T)
```

No discount-factor correction (e^{-rT} does not depend on S₀).

**Result:**

```
π_Δ = W_T / (σ S₀ T)
```

**Verification:** E[f_call · π_Δ] · disc = BS_delta ✓ (verified in tests/test_malliavin_weights.py)

**BEL connection:** For GBM, Y_s = ∂S_s/∂S₀ = S_s/S₀ and σ_diff(S) = σS, so

```
Y_s/σ_diff(S_s) = (S_s/S₀)/(σS_s) = 1/(σS₀)   (constant!)
∫₀ᵀ 1/(σS₀) dW_s = W_T/(σS₀)   →   π_Δ = W_T/(σS₀T)   ✓
```

---

## 3. Gamma (∂²V/∂S₀²)

**Derivation:** Differentiate π_Δ = W_T/(σS₀T) w.r.t. S₀, remembering W_T depends on S₀:

```
∂π_Δ/∂S₀ = (∂W_T/∂S₀)/(σS₀T) + W_T · ∂(1/(σS₀T))/∂S₀
           = [−1/(σS₀)]/(σS₀T) + W_T · (−1/(σS₀²T))
           = −1/(σ²S₀²T) − W_T/(σS₀²T)
```

The second-order Malliavin weight is π_Γ = (π_Δ)² + ∂π_Δ/∂S₀:

```
π_Γ = W_T²/(σ²S₀²T²) − 1/(σ²S₀²T) − W_T/(σS₀²T)
    = [W_T² − T − σT·W_T] / (σ²S₀²T²)
    = [W_T(W_T − σT) − T] / (σ²T²S₀²)
```

**Result:**

```
π_Γ = [W_T(W_T − σT) − T] / (S₀² σ² T²)
```

**Common mistake:** Omitting the ∂W_T/∂S₀ term gives the (wrong) formula (W_T²−T)/(S₀²σ²T²). Both have the same expectation (since E[W_T] = 0), but the correct formula is needed for unbiased path-by-path estimation — the simplified Hermite form (W_T²−T) is biased as a Monte Carlo estimator.

---

## 4. Vega (∂V/∂σ)

**Derivation:** Holding S_T (hence log(S_T/S₀)) fixed, both W_T and μ depend on σ:

```
W_T(σ) = (log(S_T/S₀) − (r−q−σ²/2)T) / σ
```

Differentiating (quotient rule, ∂(−μT)/∂σ = ∂(σ²T/2)/∂σ = σT):

```
∂W_T/∂σ = σT/σ − W_T/σ = T − W_T/σ = (σT − W_T)/σ
```

Score (∂/∂σ log p):

```
∂/∂σ log p = −(W_T/T) · ∂W_T/∂σ − 1/σ
           = −(W_T/T) · (T − W_T/σ) − 1/σ
           = −W_T + W_T²/(σT) − 1/σ
           = (W_T² − 1/T·... 
```

Collecting terms (the −1/σ from −∂log σ/∂σ):

```
∂/∂σ log p = W_T²/(σT) − W_T − 1/σ
           = (W_T² − T)/(σT) − W_T
```

No discount-factor correction (e^{-rT} does not depend on σ).

**Result:**

```
π_v = (W_T² − T)/(σT) − W_T
```

Equivalently: (W_T² − σT·W_T − T)/(σT).

---

## 5. Rho (∂V/∂r)

**Derivation:** r enters through (1) the drift μ = r−q−σ²/2 and (2) the discount factor e^{-rT}.

Score contribution from drift (∂W_T/∂r = −T/σ from ∂μ/∂r = 1 → ∂(−μT)/∂r = −T):

```
∂W_T/∂r = −T/σ   →   (−W_T/T)·(−T/σ) = W_T/σ
```

Discount contribution (∂e^{-rT}/∂r = −Te^{-rT}):

```
discount correction: −T
```

**Result:**

```
π_ρ = W_T/σ − T
```

**Verification:** For f = 1 (zero-coupon bond), Greek = disc·E[π_ρ] = disc·E[W_T/σ − T] = disc·(0 − T) = −T·disc ✓ (∂/∂r [e^{-rT}] = −T e^{-rT}).

---

## 6. Theta (∂V/∂T, convention: −∂V/∂T)

**Derivation:** T enters through (1) the log-density, (2) the drift μT, and (3) the discount e^{-rT}.

Differentiating W_T(T) with T fixed as the clock (∂W_T/∂T from ∂(−μT)/∂T = −μ, and normalisation):

```
∂W_T/∂T = −μ/σ + (W_T)/(2T)   ... but via score-function:
```

From log p ∝ −W_T²/(2T) − ½log T and W_T depending on T through the drift:

```
∂/∂T log p = W_T²/(2T²) − 1/(2T) + (W_T/T)·(μ/σ)
```

(the +μ/σ term arises from ∂W_T/∂T = μ/σ since ∂(−μT)/∂T = −μ and W_T = (log(S_T/S₀)−μT)/σ).

Theta = −∂V/∂T. Including the discount factor derivative ∂e^{-rT}/∂T = −re^{-rT} gives the additional +r term:

```
π_θ = r + (T − W_T²)/(2T²) − μ·W_T/(σT)
```

**Verification:** For f = 1 (zero-coupon bond), E[π_θ] = r + 0 − 0 = r, so Theta_ZCB = r·disc ✓.

---

## 7. Path-Dependent Delta (GBM, any payoff)

For a payoff f(S_{t₁}, ..., S_{t_n}) with fixing dates t₁ < ... < t_n = T under GBM, the joint density factorises:

```
p(S_{t₁},...,S_{t_n} | S₀) = p(S_{t₁}|S₀) · Π_{i≥2} p(S_{t_i}|S_{t_{i-1}})
```

Only the first factor depends on S₀:

```
∂/∂S₀ log p = ∂/∂S₀ log p(S_{t₁}|S₀) = ΔW₁/(σS₀Δt₁)
```

where ΔW₁ = W_{t₁} (first Brownian increment) and Δt₁ = T/n_steps (uniform grid).

**Result:**

```
π_Δ^{path} = ΔW₁ / (σ S₀ Δt₁)
```

For n_steps = 1 (t₁ = T): recovers π_Δ = W_T/(σS₀T) ✓

**Note:** The formula W_T/(σS₀T) is biased for n_steps > 1 path-dependent payoffs; it underestimates by the ratio Δt₁/T = 1/n_steps relative to the correct formula.

---

## 8. BEL Formula Derivation (Sketch)

**Theorem (Bismut–Elworthy–Li):** For dX_t = b(X_t)dt + σ(X_t)dW_t with X₀ = x:

```
∂/∂x E[f(X_T)] = (1/T) E[f(X_T) · ∫₀ᵀ (Y_s/σ(X_s)) dW_s]
```

**Proof sketch:**

(1) Write f(X_T) = E[f(X_T)] + ∫₀ᵀ Z_s dW_s (martingale representation).  
(2) Differentiate w.r.t. x under E[·]: ∂/∂x E[f(X_T)] = ∂/∂x E[∫₀ᵀ Z_s dW_s].  
(3) Applying the IBP duality: E[δ(u)·G] = E[⟨DG, u⟩_H]:

```
∂/∂x E[f(X_T)] = (1/T) E[∫₀ᵀ D_s f(X_T) · (Y_s/σ(X_s)) ds]
```

(4) Since D_s X_T = Y_s · σ(X_s)^{-1} · Y_s (via the chain rule and variational SDE):

```
D_s f(X_T) = f'(X_T) · D_s X_T
```

(5) The BEL weight ∫₀ᵀ Y_s/σ(X_s) dW_s absorbs the f'(X_T) term via the IBP, yielding the formula without f'.

The **localisation freedom** is: replace 1/T (uniform kernel) with any u_s satisfying ∫₀ᵀ u_s ds = 1. The optimal localisation (minimum weight variance) concentrates u near the times where Y_s/σ(X_s) has smallest variance.

---

## 9. Heston Delta Weight

Under Heston: dS_t = (r−q)S_t dt + √V_t S_t dW^S_t  
The first variation Y_t = ∂S_t/∂S₀ satisfies the variational SDE:

```
dY_t = (r−q + ρξ√V_t) Y_t dt + √V_t Y_t dW^S_t
```

The BEL integrand Y_t/σ_eff(S_t) = Y_t/(√V_t · S_t).  
Using the approximation Y_t ≈ S_t/S₀ (leading order, exact for ρ = 0):

```
Y_t/(√V_t S_t) ≈ (S_t/S₀)/(√V_t S_t) = 1/(S₀√V_t)
```

**BEL weight:**

```
π_Δ^{Heston} = (1/(S₀T)) Σᵢ (1/√V_{tᵢ}) ΔW^S_i
```

**Variance note:** When V_t is small (near zero), 1/√V_t is large. The Heston weight has much higher variance than GBM — the model complexity (stochastic vol) increases estimator noise. This is the price of generality.

---

## 10. Heston Vega Weight (∂V/∂V₀)

The sensitivity ∂S_t/∂V₀ requires solving a coupled variational SDE for (∂S_t/∂V₀, ∂V_t/∂V₀).  
Approximating ∂V_t/∂V₀ ≈ exp(−κt) (exact for ξ = 0):

```
π_v^{Heston} = (1/(2S₀T)) Σᵢ exp(−κtᵢ)/√V_{tᵢ} · ΔW^S_i
```

The factor 1/2 arises from ∂σ_eff/∂V₀ = 1/(2√V₀).

---

## 11. Second-Order Weights (Gamma, Vanna, Volga)

### 11.1 GBM Gamma (revisited as BEL)

The second-order BEL weight applies IBP twice. Define the first-order kernel K₁ = W_T/(σS₀).  
The gamma weight satisfies:

```
π_Γ = K₁ · π_Δ + ∂K₁/∂S₀
```

Since K₁ = W_T/(σS₀) and ∂K₁/∂S₀ = (∂W_T/∂S₀)/(σS₀) + W_T·(−1/(σS₀²)):

```
π_Γ = [W_T(W_T − σT) − T]/(σ²T²S₀²)   ✓
```

### 11.2 Vanna (∂²V/∂S₀∂σ)

```
π_vanna = π_Δ · π_v + ∂π_v/∂S₀
```

where π_Δ = W_T/(σS₀T) and π_v = (W_T²−T)/(σT) − W_T.

Since ∂π_v/∂S₀ = (2W_T·∂W_T/∂S₀)/(σT) − ∂W_T/∂S₀  
and ∂W_T/∂S₀ = −1/(σS₀):

```
π_vanna = [W_T/(σS₀T)] · [(W_T²−T)/(σT) − W_T]
         − [2W_T/(σT) − 1]/(σS₀)
```

### 11.3 Volga (∂²V/∂σ²)

```
π_volga = π_v² + ∂π_v/∂σ
```

where π_v = (W_T²−T)/(σT) − W_T and ∂W_T/∂σ = T − W_T/σ:

```
∂π_v/∂σ = −(W_T²−T)/(σ²T) + [2W_T/(σT)]·(T − W_T/σ) − (T − W_T/σ)
         = −(W_T²−T)/(σ²T) + 2W_T/σ − 2W_T²/(σ²T) − T + W_T/σ
```

This simplifies to a polynomial in W_T of degree 4 divided by σ²T. All terms are in closed form and implemented in `mgreeks/greeks/malliavin.py:higher_order()`.
