# S3b.1 — systematic residual characterization

Overlay of target-under-thin (shipped thin-B.3.5 FA kinetics) vs Fournier 2000
Fig 6A Déa observations (n=9–15). Produced by `s3b1_overlay_fournier_dea.py`.

## Per-rank fit quality at optimal per-rank offset

| rank | obs peak (cm) | sim peak (cm) | best Δ (°Cd) | RMSE@Δ (cm) | RMSE@0 (cm) |
|------|---------------|---------------|--------------|-------------|-------------|
|   9  | 22.0          | 23.0          | +216         | **1.02**    | 18.10       |
|  10  | 22.0          | 22.0          | +250         | **0.79**    | 18.27       |
|  11  | 22.0          | 22.0          | +280         | **0.93**    | 17.69       |
|  12  | 21.0          | 20.0          | +319         | **0.98**    | 15.22       |
|  13  | 16.5          | 18.0          | +340         | **1.29**    | 11.22       |
|  14  | 12.0          | 18.9          | +360         | **1.88**    |  8.15       |
|  15  |  9.0          | 11.3          | +398         | **1.12**    |  5.94       |

Mean Δ = **+309 ± 59 °Cd**. Per-rank RMSE at optimal offset spans 0.79–1.88 cm,
well within the ±2 cm band (±1 cm digitization + ~1 cm residual). **Curve shape
is matched.**

## Finding 1: monotone plastochron drift (load-bearing)

Per-rank Δ increases monotonically from +216 at rank 9 to +398 at rank 15 —
a +182 °Cd spread across 7 ranks, or **~30 °Cd per rank of extra plastochron
in the sim relative to Fournier Déa**:

| rank transition | ΔΔ (°Cd) |
|-----------------|----------|
| 9 → 10          | +34      |
| 10 → 11         | +30      |
| 11 → 12         | +39      |
| 12 → 13         | +21      |
| 13 → 14         | +20      |
| 14 → 15         | +38      |

Observed Déa primordium-initiation rate is 0.044 °Cd⁻¹ (Fournier 2000 §Results
p. 558), i.e. **~23 °Cd plastochron**. Sim's effective plastochron on the
Andrieu axis is 23 + 30 = **~53 °Cd (2.3× Déa)**.

**Root cause**: leaf emergence times in `maize_calibrated.xml` gate on the
legacy Tb=8 Nielsen axis (parent plan §5 — "leaves gate on Tb=8 axis,
independent of FA kinetics"). The Nielsen V-stage calendar for Juelich maize
is slower than Déa's Grignon conditions, so when re-expressed on the Andrieu
Tb=9.8 axis (which drives `calcLengthPerPhytomer`), the spacing between
successive rank-primordium initiations is roughly double the Déa value.

**Implication for S3b.5**: a shared-TT_A-axis per-rank overlay against
Fournier will NOT land within ±15% because the curves are time-shifted
relative to each other with a cultivar-inherent +30 °Cd/rank drift. Two
options:

1. **Per-rank τ_n-axis comparison.** Express both curves in τ_n = TT_A −
   primordium_init_TT_n coordinates. This collapses the plastochron drift by
   construction and tests only the *shape* of FA kinetics. Clean pass at ±15%
   fully expected (current S3b.1 RMSE 0.79–1.88 cm ≈ 5–10% of peak IL).

2. **Shared-TT axis with documented residual.** Retain absolute-TT overlay but
   widen per-rank tolerance to ±25% OR ±3 cm and prefix the writeup claim with
   "given the Nielsen-axis leaf calendar." Weaker claim but no coordinate
   gymnastics.

**Recommendation**: S3b.5 adopt option 1 (τ_n-axis). It validates exactly
what S3b.2 is introducing (per-phytomer geometric embedding of the FA shape)
without entangling it with a known Ch1-inherited calendar residual. The
absolute-TT overlay in this file (Ch2 Fig Xc candidate) can still ship as a
supporting figure with the plastochron drift annotated.

## Finding 2: Phase IV decay artifact (target drops)

`calcLengthPerPhytomer` Phase IV interpolates from `IL_end_III` down to
`IL_final`; when `IL_end_III > IL_final` (low ranks where `v_n·D_n`
overshoots), the returned target *decreases*. Under thin-B.3.5 this is
harmless (the scalar `max(·, calcLength(age))` floor keeps total stem length
monotonic); under S3b.2's pure Σ IL driver it matters. Decision 2
(per-rank monotonic latch `length_per_n[n] += max(0, dl_n)`) handles it.

Visible in the left panel of `s3b1_thin_target_per_rank.png`: ranks 5–12
peak at ~18–23 cm then decay toward ~16 cm. Fournier observations are
monotonic so this artifact does NOT appear in the overlay — but **S3b.5
must validate against the latched realisation `length_per_n[n]`, not against
the raw `calcLengthPerPhytomer(n)` output.**

## Finding 3: rank-14 amplitude overshoot (minor)

Sim rank 14 peaks at 18.9 cm vs Fournier observation's final visible value of
12.0 cm. Sim asymptote is ~18 cm; Fournier data end at TT=640 °Cd where the
curve is still rising. Birch 2002 Fig 6a Déa static data (final lengths) show
rank-14 Déa final length of ~15 cm. Sim over-predicts by ~3 cm (~20%). This
is within the phase_III parameter uncertainty per the `D_n` cross-check
ranges in `phase_III_per_rank.json` and doesn't warrant recalibration for
S3b.

## Finding 4: baseline established for Ch2 figures

The two files ready for the FSPM 2026 / Ch2 methods section:

- **Fig Xa (target-under-thin)**: `s3b1_thin_target_per_rank.png`, left panel —
  the kinetic target curves per rank as computed by `calcLengthPerPhytomer`
  under thin-B.3.5. Shows the sigmoidal FA shapes and the Phase IV decay that
  S3b.2's monotonic latch resolves.
- **Fig Xc (target vs observed)**: `s3b1_overlay_fournier_dea.png` — per-rank
  panels showing shape match (±2 cm, mean RMSE 1.14 cm) after per-rank
  axis alignment. Plastochron drift annotated.

Fig Xb (S3b.2's *achieved* H_n(τ_n) curve post-per-phytomer embedding) will
be produced by S3b.5 on the same τ_n axis recommended above.

## Recommendation for S3b.5 test tolerance

Two thresholds:

1. **Primary test (must pass)**: per-rank RMSE of achieved IL_n(τ_n) vs
   digitized Fournier ≤ **2.5 cm** at every rank 9–15. (Current S3b.1 target
   curves: 0.79–1.88 cm; S3b.2's per-phytomer latch should reduce noise
   further by eliminating the Phase IV decay artifact → expect 0.8–2.0 cm.)

2. **Secondary test (quality flag, non-blocking)**: per-rank absolute-axis
   residual ≤ **±25%** of observed IL_n at each Fournier sample point after
   applying the mean Δ = +309 °Cd offset. Cross-rank Δ spread documented in
   the writeup, not asserted against a threshold.

Skip the ±15% test the plan templated — that was assuming zero plastochron
drift; this pre-work shows there's a +30 °Cd/rank drift that's structural
(Nielsen calendar), not something S3b.2 can fix.

## Data files

- `s3b1_thin_target_per_rank.json` — sim-side kinetic targets, 130 daily samples, ranks 1–16
- `fournier2000_dea_fig6a_per_rank.json` — observed Déa IL_n(TT), ranks 9–15, manually digitised from `_page_5_Figure_6.jpeg` (Fournier 2000 Fig 6A)
- `s3b1_thin_target_per_rank.png` — Ch2 Fig Xa candidate (target-only view)
- `s3b1_overlay_fournier_dea.png` — Ch2 Fig Xc candidate (target vs observed, plastochron drift annotated)
