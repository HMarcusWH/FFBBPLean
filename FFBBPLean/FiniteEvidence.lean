import Mathlib
import PermanssonLean.Probability.FinitePrefixTV

/-!
# FFBBP: finite evidence and posterior-enclosure fundamentals

**Scope**: finite candidate histories with rational weights, finite sums,
and strictly-positive normalizing evidence. This theorem does *not* assert
that FFBBP's full positive-history Janossy density is normalized or that
Gaussian integration has been certified. Its only purpose is to prevent
invalid interval-to-posterior promotion at the discrete inference boundary.
-/

namespace FFBBPLean

variable {ι : Type*} [DecidableEq ι]

/-- Candidate-wise intervals enclose the total finite unnormalized evidence. -/
theorem finite_evidence_interval
    (s : Finset ι) (lower actual upper : ι → ℚ)
    (hLower : ∀ i ∈ s, lower i ≤ actual i)
    (hUpper : ∀ i ∈ s, actual i ≤ upper i) :
    (∑ i ∈ s, lower i) ≤ (∑ i ∈ s, actual i) ∧
    (∑ i ∈ s, actual i) ≤ (∑ i ∈ s, upper i) := by
  constructor
  · exact Finset.sum_le_sum hLower
  · exact Finset.sum_le_sum hUpper

/-- Fail-closed finite evidence normalization: a numerator and a
positive denominator with admitted enclosing bounds give enclosing
bounds on the resulting posterior expectation. -/
theorem normalized_nonnegative_ratio_interval
    (zLower zUpper nLower nUpper z n : ℚ)
    (hzLower : 0 < zLower)
    (hzLo : zLower ≤ z) (hzHi : z ≤ zUpper)
    (hnLower : 0 ≤ nLower)
    (hnLo : nLower ≤ n) (hnHi : n ≤ nUpper) :
    nLower / zUpper ≤ n / z ∧ n / z ≤ nUpper / zLower := by
  have hz : 0 < z := lt_of_lt_of_le hzLower hzLo
  have hzUpper : 0 < zUpper := lt_of_lt_of_le hz hzHi
  have hnUpper : 0 ≤ nUpper := le_trans hnLower (le_trans hnLo hnHi)
  constructor
  · apply (div_le_div_iff₀ hzUpper hz).2
    have h₁ : nLower * z ≤ nLower * zUpper :=
      mul_le_mul_of_nonneg_left hzHi hnLower
    have h₂ : nLower * zUpper ≤ n * zUpper :=
      mul_le_mul_of_nonneg_right hnLo (le_of_lt hzUpper)
    exact le_trans h₁ h₂
  · apply (div_le_div_iff₀ hz hzLower).2
    have h₁ : n * zLower ≤ nUpper * zLower :=
      mul_le_mul_of_nonneg_right hnHi (le_of_lt hzLower)
    have h₂ : nUpper * zLower ≤ nUpper * z :=
      mul_le_mul_of_nonneg_left hzLo hnUpper
    exact le_trans h₁ h₂

/-- A concrete finite exact-rational posterior-normalization smoke test. -/
example : (1 : ℚ) / 5 ≤ (2 : ℚ) / 6 ∧ (2 : ℚ) / 6 ≤ (3 : ℚ) / 4 := by
  constructor <;> norm_num

end FFBBPLean
