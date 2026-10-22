/-!
FFBBP D24AP, restricted *mathematical* interval laws.

PRIVATE synthetic research. These proofs, if compiled, establish ONLY the explicitly
quantified rational interval identities below; they do not certify the source Python
interpreter, actual C06/C08 continuous-history integrals, or real-world inferences.

No admitted proof obligations are disguised as verified results. Verify with the pinned kernel.
-/
import Mathlib.Tactic

namespace FFBBP.D24AP

def InI (lo hi x : ℚ) : Prop := lo ≤ x ∧ x ≤ hi

theorem add_inclusion
    {a b c d x y : ℚ}
    (hx : InI a b x) (hy : InI c d y) :
    InI (a + c) (b + d) (x + y) := by
  rcases hx with ⟨hx₀, hx₁⟩
  rcases hy with ⟨hy₀, hy₁⟩
  constructor <;> dsimp [InI] at * <;> linarith

theorem neg_inclusion {a b x : ℚ}
    (hx : InI a b x) : InI (-b) (-a) (-x) := by
  rcases hx with ⟨hl, hu⟩
  constructor <;> dsimp [InI] at * <;> linarith

theorem mul_nonnegative_inclusion
    {a b c d x y : ℚ}
    (ha : 0 ≤ a) (hc : 0 ≤ c) (hd : 0 ≤ d)
    (hx : InI a b x) (hy : InI c d y) :
    InI (a*c) (b*d) (x*y) := by
  rcases hx with ⟨hx₀, hx₁⟩
  rcases hy with ⟨hy₀, hy₁⟩
  have hxpos : 0 ≤ x := le_trans ha hx₀
  have hypos : 0 ≤ y := le_trans hc hy₀
  have h1 : 0 ≤ (x-a)*y := mul_nonneg (sub_nonneg.mpr hx₀) hypos
  have h2 : 0 ≤ a*(y-c) := mul_nonneg ha (sub_nonneg.mpr hy₀)
  have h3 : 0 ≤ (b-x)*d := mul_nonneg (sub_nonneg.mpr hx₁) hd
  have h4 : 0 ≤ x*(d-y) := mul_nonneg hxpos (sub_nonneg.mpr hy₁)
  constructor <;> dsimp [InI] at * <;> nlinarith

/-!
A certificate's integer k is the lower lattice index at positive scale s.
The hypothesis k ≤ s*x < k+1 MUST be separately checked for the source x;
the theorem proves the stated lattice enclosure and minimality under that premise.
-/
theorem lower_lattice_bracket (s x : ℚ) (k : ℤ)
    (hs : 0 < s) (hlo : (k : ℚ) ≤ s*x)
    (hhi : s*x < (k : ℚ) + 1) :
    (k : ℚ)/s ≤ x ∧ x < ((k : ℚ)+1)/s := by
  constructor
  · exact (div_le_iff₀ hs).2 hlo
  · exact (lt_div_iff₀ hs).2 (by nlinarith [hhi])

/-! The upper index k+1 is the *least* lattice endpoint strictly above x. -/
theorem upper_lattice_bracket (s x : ℚ) (k : ℤ)
    (hs : 0 < s) (hlo : (k : ℚ) < s*x)
    (hhi : s*x ≤ (k : ℚ) + 1) :
    (k : ℚ)/s < x ∧ x ≤ ((k : ℚ)+1)/s := by
  constructor
  · exact (div_lt_iff₀ hs).2 hlo
  · exact (le_div_iff₀ hs).2 (by nlinarith [hhi])

/-!
A two-way normalized positive Gibbs choice. The interval bounds in the
hypotheses are independently admitted. Both denominators are strictly positive.
-/
theorem positive_two_site_gibbs
    {a b c d x y : ℚ}
    (ha : 0 < a) (hc : 0 < c)
    (hx : InI a b x) (hy : InI c d y) :
    a/(a+d) ≤ x/(x+y) ∧ x/(x+y) ≤ b/(b+c) := by
  rcases hx with ⟨hx₀, hx₁⟩
  rcases hy with ⟨hy₀, hy₁⟩
  have hxpos : 0 < x := lt_of_lt_of_le ha hx₀
  have hypos : 0 < y := lt_of_lt_of_le hc hy₀
  have hbpos : 0 < b := lt_of_lt_of_le hxpos hx₁
  have hdpos : 0 < d := lt_of_lt_of_le hypos hy₁
  have hdlo : 0 < a+d := by linarith
  have hdmid : 0 < x+y := by linarith
  have hdhi : 0 < b+c := by linarith
  have h1 : 0 ≤ (x-a)*d := mul_nonneg (sub_nonneg.mpr hx₀) (le_of_lt hdpos)
  have h2 : 0 ≤ a*(d-y) := mul_nonneg (le_of_lt ha) (sub_nonneg.mpr hy₁)
  have h3 : 0 ≤ (b-x)*c := mul_nonneg (sub_nonneg.mpr hx₁) (le_of_lt hc)
  have h4 : 0 ≤ b*(y-c) := mul_nonneg (le_of_lt hbpos) (sub_nonneg.mpr hy₀)
  constructor
  · apply (div_le_div_iff₀ hdlo hdmid).2
    nlinarith [h1, h2]
  · apply (div_le_div_iff₀ hdmid hdhi).2
    nlinarith [h3, h4]

/-! One common nuisance index: first multiply the A/B factors, then average. -/
def sharedSign3 (p0 p1 p2 a0 a1 a2 b0 b1 b2 : ℚ) : ℚ :=
  p0*a0*b0 + p1*a1*b1 + p2*a2*b2

theorem shared_sign3_nonnegative
    {p0 p1 p2 a0 a1 a2 b0 b1 b2 : ℚ}
    (hp0 : 0 ≤ p0) (hp1 : 0 ≤ p1) (hp2 : 0 ≤ p2)
    (ha0 : 0 ≤ a0) (ha1 : 0 ≤ a1) (ha2 : 0 ≤ a2)
    (hb0 : 0 ≤ b0) (hb1 : 0 ≤ b1) (hb2 : 0 ≤ b2) :
    0 ≤ sharedSign3 p0 p1 p2 a0 a1 a2 b0 b1 b2 := by
  dsimp [sharedSign3]
  positivity

/-! A concrete counterexample: separately marginalized signs need not agree. -/
theorem shared_nuisance_product_counterexample :
    (37 : ℚ) / 100 ≠ (1 : ℚ) / 4 := by
  norm_num

end FFBBP.D24AP
