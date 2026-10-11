# FFBBP B16: claim boundary and independent mathematical obligations

## Immutable parents

The complete FFBBP v2.0.1 private release of October 10, 2026 remains an
independent frozen archive. Its v1.7 A0/RUN46F evaluation is historical and
**does not** validate the finite Bayesian model. The original B15 strong-coupling
NO-GO width was ~0.875669 against the <=0.001 requirement. This work introduces
B16 as a separately identifiable numerical candidate, not a retroactive result.

## Mathematical object being evaluated

T=3, D=2 and one observation per scan; Poisson births, Markov object survival
and motion, detection, independent Poisson clutter, observed Gaussian marks,
with a Gaussian AR(1) field modulating birth/survival/motion/detection and
continuous mark emissions. The candidate computes the conditional Janossy
likelihood using positive physical-history track intensities and a
first-detection normalization. Its numerical input digest must match the
frozen B15 input before acceptance testing.

## Derivation required to promote the B16 result

1. **Source-to-model equivalence.** Prove that the runtime conditional law
   enumerates *all and only* permitted finite physical histories and disjoint
   observation origins, and that the Poisson `exp(-λ)` factor accounts for
   no-detection histories without double counting. Independently compare
   Reference A and Reference B at selected field points and on controlled
   input families; program agreement is not itself a theorem.
2. **Finite Gaussian-cell calculus.** The independent innovations Z_j have
   standard normal laws. For each partition cell, derive exact cell mass,
   truncated conditional means and variances. The code's exact-rational
   implementation of π, sqrt, exp, Φ and truncated moments depends on
   explicitly stated Taylor/alternating-series remainder inequalities.
3. **Derivative coverage.** Independently prove the interval Jacobian and
   Hessian bounds for birth-rate exp, logistic survival/detection, the two-state
   softmax motion row, Gaussian emission, track products, the PPP normalizer,
   and every sum/product carried by the Janossy DP. All local boxes must be
   convex and must cover their full conditional affine-Gaussian support.
4. **Evidence and numerator remainder.** Under local derivative bounds,
   use centered Taylor and Cauchy-Schwarz bounds on the conditional evidence
   and the signed first-moment numerators; accumulated rounding must be
   directed outward. No unbounded/unproven ratio is permitted.
5. **Gaussian tails.** Prove the innovation-coordinate union bound and the
   global positive likelihood bound, plus a second-moment Cauchy bound for
   signed first moments outside the clipped box.
6. **Executable arithmetic and representation.** Validate the Python Decimal
   version, outward padding of exp/sqrt, exact binary-float interpretation
   of frozen model parameters, and cache/process deterministic behavior.
   Independently cross-check with a separate validated interval backend
   (e.g., Arb/MPFI) before any mathematical certification claim.
7. **Formal bridge.** Mechanize the likelihood recursion / independent
   complete-history equality, the finite and Gaussian bounds, their
   prerequisites and the emitted numeric witness in Lean, pinned to source
   and benchmark digests. The present `FFBBPLean/FiniteEvidence.lean` proves
   *only* elementary finite weight aggregation and nonnegative ratios.

## Currently demonstrable vs open

| Claim | Current status |
|---|---|
| Directed decimal B16 output is locally reproducible | Executable evidence; require fresh CI receipt |
| B16 emitted widths numerically meet the original frozen thresholds | Conditional pass, **not** a mathematical certification |
| Finite evidence/normalizer ratio algebra | Lean target, subject to CI |
| Complete FFBBP Janossy likelihood Lean proof | OPEN |
| Gaussian transcendentals and derivative bounds Lean proof | OPEN |
| Independently validated floating implementation | OPEN |
| General PMBM/GLMB closure or arbitrary-cardinality efficiency | OPEN |
| External blind validation / real-world causal attribution | NOT ESTABLISHED |
| Privacy/operational deployment qualification | NOT ESTABLISHED |

**No final FFBBP 2.0 completion/operational claim is licensed by this branch.**
