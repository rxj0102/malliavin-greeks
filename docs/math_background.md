# Mathematical Background: Malliavin Calculus for Greeks

## 1. Malliavin Derivative

### 1.1 Fréchet Derivative on Wiener Space

Let (Ω, F, P) be a Wiener space with Ω = C₀([0,T]; ℝ) (continuous paths starting at 0).  
Every ω ∈ Ω defines a Brownian motion W_t(ω) = ω(t).

The **Cameron–Martin space** H = L²([0,T]) embeds into Ω via the map  
h ↦ ∫₀ᵀ h(s) ds (the primitives of H-valued shifts).

**Definition (Malliavin derivative).** A random variable F: Ω → ℝ is **Malliavin differentiable** if there exists a process D_s F ∈ L²(Ω × [0,T]) such that for every h ∈ H:

```
lim_{ε→0} (F(ω + ε ∫₀ᵀ h(s)ds) - F(ω)) / ε = ∫₀ᵀ D_s F · h(s) ds   in L²(P)
```

The map s ↦ D_s F is the **Malliavin derivative** of F, a Fréchet derivative in the Cameron–Martin direction h.

### 1.2 Domain and Chain Rule

The domain of D is denoted **D^{1,2}**, the closure of smooth cylindrical functionals  
F = φ(W_{t₁}, ..., W_{t_n}) under the Sobolev norm:

```
‖F‖²_{D^{1,2}} = E[F²] + E[∫₀ᵀ |D_s F|² ds]
```

**Chain rule:** If F = φ(F₁, ..., F_n) with φ ∈ C¹ and F_i ∈ D^{1,2}:

```
D_s φ(F₁,...,F_n) = Σ_i (∂φ/∂x_i)(F₁,...,F_n) · D_s F_i
```

### 1.3 GBM Example

Under GBM: S_T = S_0 exp((r - q - σ²/2)T + σW_T).  
Since W_T = ∫₀ᵀ 1 dW_s, we have D_s W_T = 1 for all s ∈ [0,T].

By the chain rule:

```
D_s S_T = S_T · σ · D_s W_T = σ S_T   for all s ≤ T
```

This is remarkable: the Malliavin derivative D_s S_T is **independent of s** — a perturbation of the Brownian path at any time s ≤ T produces the same multiplicative effect σS_T on the terminal value.  This is a special property of GBM that simplifies all subsequent formulas.

More generally, for the full path S_t:

```
D_s S_t = σ S_t   for s ≤ t,   D_s S_t = 0 for s > t
```

---

## 2. Integration by Parts on Wiener Space

### 2.1 The Duality Formula

The key tool is the **duality** between D and the **Skorokhod integral** δ (also written ∫·dW in the adapted case):

```
E[F · δ(u)] = E[∫₀ᵀ D_s F · u_s ds]   for F ∈ D^{1,2}, u ∈ Dom(δ)
```

For adapted u, δ(u) = ∫₀ᵀ u_s dW_s (Itô integral); for non-adapted u, it extends to the Skorokhod integral.

**Integration by parts (IBP) on Wiener space:**  
For any F, G ∈ D^{1,2} and u ∈ Dom(δ):

```
E[G · ∫₀ᵀ u_s dW_s] = E[∫₀ᵀ D_s G · u_s ds]
```

### 2.2 The Malliavin Weights Formula

Let f: ℝ → ℝ (possibly discontinuous) and F = f(S_T) with S_T ∈ D^{1,2}.  
We want to compute ∂/∂x₀ E[f(S_T)] without differentiating f.

**Key insight:** apply the IBP formula to "move" the S₀-derivative off f and onto the weight:

```
∂/∂S₀ E[f(S_T)] = E[f'(S_T) · ∂S_T/∂S₀]        (naive, requires f')
                 = E[f(S_T) · H(G)]               (IBP, no f' needed)
```

where H(G) is the **Malliavin weight** obtained by the IBP procedure.

**Theorem (Fournié et al., 1999):** For any measurable f: ℝ → ℝ with E[f(S_T)²] < ∞:

```
∂/∂S₀ E[f(S_T)] = E[f(S_T) · π]
```

where π = (1/T) ∫₀ᵀ (Y_s / σ(S_s)) dW_s and Y_s = ∂S_s/∂S₀.

For GBM: σ(S) = σS, Y_s = S_s/S₀, so Y_s/σ(S_s) = 1/(σS₀), and:

```
π_Δ = W_T / (σ S₀ T)
```

---

## 3. Connection to Clark–Ocone Formula

### 3.1 Clark–Ocone Representation

Every F ∈ D^{1,2} with E[F²] < ∞ admits the **Clark–Ocone representation**:

```
F = E[F] + ∫₀ᵀ E[D_t F | F_t] dW_t
```

This is the Malliavin calculus analogue of the martingale representation theorem.  
The adapted projection E[D_t F | F_t] gives the hedging strategy for the functional F.

### 3.2 Link to Delta Hedging

For a discounted payoff F = e^{-rT} f(S_T), the Clark–Ocone formula gives:

```
F = E[F] + ∫₀ᵀ E[D_t F | F_t] dW_t
```

Under GBM, D_t F = e^{-rT} f'(S_T) · D_t S_T = e^{-rT} f'(S_T) · σ S_T.  
Taking the conditional expectation at t:

```
E[D_t F | F_t] = σ S_t · E[e^{-r(T-t)} f'(S_T) · S_T/S_t | F_t]
               = σ S_t · Δ_t
```

where Δ_t is the **option delta** at time t.  The Clark–Ocone formula thus recovers the Itô representation of the option price as the hedged portfolio, confirming the Malliavin framework is consistent with arbitrage pricing.

---

## 4. Why the Weight Is Independent of the Payoff

The IBP formula separates the payoff f from the weight π structurally:

```
∂/∂θ E[f(S_T)] = E[f(S_T) · π_θ]
```

The weight π_θ depends ONLY on:
- The **model dynamics** (the diffusion coefficient σ, the drift)
- The **parameter θ** being differentiated
- The **Brownian path** (via W_T or the path integral)

It does NOT depend on f. This has profound consequences:
1. **Single simulation:** compute all Greeks from one set of paths by multiplying by different weights
2. **Discontinuous payoffs:** f never appears differentiated — digitals, barriers, and other discontinuous payoffs work without modification
3. **Regime changes:** changing the payoff structure (e.g., adding a knock-out feature) does not change the weight

---

## 5. Why Discontinuous Payoffs Work

For a digital call f(S_T) = 1_{S_T > K}, the naive pathwise estimator requires:

```
∂/∂S₀ E[1_{S_T > K}] = E[δ(S_T - K) · ∂S_T/∂S₀]
```

where δ is the Dirac delta — **not a function**, and the Monte Carlo estimator has **infinite variance**.

The Malliavin IBP moves the derivative off f:

```
∂/∂S₀ E[1_{S_T > K}] = E[1_{S_T > K} · W_T/(σS₀T)]
```

Now f = 1_{S_T > K} is bounded and the weight W_T/(σS₀T) is square-integrable, so the estimator has **finite variance**. The payoff is never differentiated.

**Heuristic:** The IBP "transfers" the singularity of f' (the Dirac delta) onto the Brownian motion W, where it is smoothed by the Gaussian measure.

---

## 6. Hermite Polynomial Connection

Higher-order Greeks involve **Hermite polynomials** as weights.  
Let H_n(x) = (-1)^n e^{x²/2} d^n/dx^n e^{-x²/2} be the probabilist's Hermite polynomials:

```
H_0(x) = 1,  H_1(x) = x,  H_2(x) = x² - 1,  H_3(x) = x³ - 3x, ...
```

Under GBM, the n-th order weight (for ∂^n/∂S₀^n) is:

```
π^{(n)} = H_n(W_T/√T) / (S₀^n σ^n T^{n/2})   (up to normalization)
```

Specifically:
- **Delta (n=1):** π_Δ = H₁(Z)/(S₀σ√T) = Z/(S₀σ√T)  where Z = W_T/√T
- **Gamma (n=2):** π_Γ = H₂(Z)/(S₀²σ²T) = (Z²-1)/(S₀²σ²T) — **with correction**

The gamma correction arises because W_T itself depends on S₀ (when viewed as a function of S_T), requiring an extra term from differentiating W_T w.r.t. S₀.  The correct gamma weight is:

```
π_Γ = [W_T(W_T - σT) - T] / (S₀²σ²T²)
```

which includes the term -σT·W_T from ∂W_T/∂S₀ = -1/(σS₀).

For **mixed second-order Greeks** (vanna = ∂²V/∂S₀∂σ, volga = ∂²V/∂σ²), the weights are products of the first-order weights plus a cross-term:

```
π_vanna = π_Δ · π_v + ∂π_v/∂S₀
π_volga = π_v² + ∂π_v/∂σ
```

(both computed in closed form under GBM via the chain rule on the log-density).

---

## 7. The Bismut–Elworthy–Li Formula

For general diffusions dX_t = b(X_t)dt + σ(X_t)dW_t with X₀ = x, the **BEL formula** provides:

```
∂/∂x E[f(X_T)] = (1/T) E[f(X_T) · ∫₀ᵀ (Y_s/σ(X_s)) dW_s]
```

where Y_s = ∂X_s/∂x is the **first-variation process** satisfying:

```
dY_s = b'(X_s) Y_s ds + σ'(X_s) Y_s dW_s,   Y₀ = 1
```

The integrand Y_s/σ(X_s) is the **ratio of the first variation to the diffusion coefficient** — it measures how sensitive the Brownian driving noise at time s is to the initial condition x.

**GBM special case:** σ(S) = σS, b(S) = (r-q)S, so Y_s = S_s/S₀ and  
Y_s/σ(S_s) = (S_s/S₀)/(σS_s) = 1/(σS₀) (constant!), giving W_T/(σS₀T) as before.

**General case (numerical BEL):** For local vol σ(t,S) and Heston √V·S, the integral must be computed numerically by stepping through the path and accumulating (Y_{t_i}/σ(X_{t_i}))·ΔW_i.

---

## 8. References

1. **Nualart, D. (2006)** — *The Malliavin Calculus and Related Topics*, 2nd ed. Springer.  
   The standard reference for the mathematical foundations.

2. **Fournié, E., Lasry, J.-M., Lebuchoux, J., Lions, P.-L. & Touzi, N. (1999)** —  
   "Applications of Malliavin calculus to Monte Carlo methods in finance."  
   *Finance and Stochastics* **3**, 391–412.  
   First systematic application to option Greeks; derives the score-function weights for GBM.

3. **Fournié, E., Lasry, J.-M., Lebuchoux, J. & Lions, P.-L. (2001)** —  
   "Applications of Malliavin calculus to Monte Carlo methods in finance II."  
   *Finance and Stochastics* **5**, 201–236.  
   Extension to stochastic volatility, path-dependent payoffs, and variance reduction.

4. **Bismut, J.-M. (1984)** — *Large Deviations and the Malliavin Calculus*. Birkhäuser.  
   Original probabilistic proof of the integration by parts formula on path space.

5. **Elworthy, K.D. & Li, X.-M. (1994)** — "Formulae for the derivatives of heat semigroups."  
   *J. Functional Analysis* **125**, 252–286.  
   Geometric version of the BEL formula for diffusions on manifolds.

6. **Gobet, E. & Kohatsu-Higa, A. (2003)** — "Computation of Greeks for barrier and  
   look-back options using Malliavin calculus." *Electronic Communications in Probability* **8**, 51–62.  
   Localization techniques for barrier options to reduce weight variance.

7. **Glasserman, P. (2004)** — *Monte Carlo Methods in Financial Engineering*. Springer.  
   Chapter 7 covers pathwise and likelihood-ratio methods; provides the comparison framework  
   for evaluating Malliavin vs alternative estimators.
