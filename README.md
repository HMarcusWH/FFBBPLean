# FFBBPLean — private B16 numerical research and scoped Lean development

**PRIVATE / INTERNAL — NOT FOR PUBLICATION OR OPERATIONAL USE**

This repository contains the **new experimental B16 branch** for the original
frozen strongly-coupled three-scan FFBBP numerical acceptance target, plus a
separate formalization research target. It is deliberately **not** the FFBBP
v2.0.1 release; the historical v1.7/v2.0/v2.0.1 archives and acceptance results
must not be changed or silently superseded.

## Current achieved result (conditional)

The B16 directed-decimal finite Gaussian-cell computation reports, for the
**unchanged** T=3, D=2, N=3 input digest `7004149342c03dd395ed30a35775576553ff9c091c1f0fedb90e610e9a9aa885`:

| Gate | Precommitted target | B16 conditional result |
|---|---:|---:|
| Gaussian-support leaves | ≤12,000 | 5,832 |
| Evidence interval width | ≤0.001 | 0.00069539553 |
| Largest posterior-mean width | ≤1.0 | 0.40328646 |

These are **conditional mathematical enclosures**, not fully certified results.
The current code uses exact-rational normal CDF cell bounds and outward-rounded
Decimal operations, with analytically motivated derivative bounds. Those bounds,
source-law correspondence, and library semantics still require independent review
and mechanized verification. Python tests and CI are reproducibility evidence,
**not proof** that the entire FFBBP solver is correct.

## Reproduce

Requires Python 3.13. From the repository root:

```sh
python -m pip install -r requirements.txt
PYTHONPATH=. python -m pytest -q tests/
PYTHONPATH=. python run_directed.py 18
```

The default 8-bin run is a diagnostic; specify **18** to reproduce the original
conditional numerical target. Output: `receipts/b16_decimal_directed_n18.json`.
The frozen model is self-contained in `ffbbp_v198b/`; References A and B are
independent algorithmic constructions. `b16/gaussian_cells.py` and
`b16/hessian_cells.py` are earlier **uncertified exploratory** implementations;
only `b16/directed_certificate.py` is the directed-decimal experiment.

## Formalization

The separate Lean target pins [PermanssonLean](https://github.com/HMarcusWH/PermanssonLean)
at commit `821d8545ce0ef28466d4aba70c415db6953f5cef`. It begins with
**finite rational evidence enclosure and normalization**; it does **not** yet
mechanize the full finite Janossy law, Gaussian CDF remainder inequalities,
Taylor derivative global bounds, numeric Decimal implementation, or complete
Bayesian posterior. These remain open gates.

```sh
lake exe cache get
lake build FFBBPLean
```

See `docs/ASSURANCE_SCOPE.md` for precise mathematical and claim boundaries.
