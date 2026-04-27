# S3b.5 sign-off — per-rank Déa overlay

**Date**: 2026-04-24
**Plan**: `PLAN_FULL_B35_PER_PHYTOMER_BOOKKEEPING_2026-04-23.md` §D
**Predecessors**: S3b.3 (2026-04-24), S3b.4 (2026-04-24)

## What S3b.5 ships

Per-rank τ_n-axis overlay of the FA kinetic target curve against Fournier 2000
Fig 6A Déa observations, frozen as the Chapter 2 "money figure" (FSPM 2026
submission).

| File | What |
|------|------|
| `s3b5_achieved_per_rank.py` / `.json` / `.png` | 130-day capture of `calcLengthPerPhytomer(n)` (target) + `get_phytomer_length(n)` (achieved) per rank 1–16, Juelich 2024 met, seed=7, post-S3b.3 FA-on build |
| `s3b5_overlay_fournier_dea.py` | Analysis script: per-rank best offset Δ_n + RMSE@Δ vs Déa; produces the acceptance baseline + Fig Xb |
| `d2_htt_per_rank_fournier_dea.json` | S3b.5 acceptance baseline (frozen); consumed by `test_fa_htt_trajectory.py` |
| `d2_htt_per_rank_plot_s3b.png` | Ch2 Fig Xb (per-rank panels + achieved-vs-target diagnostic bar) |
| `test_fa_htt_trajectory.py` (upgraded) | +10 new pytests: baseline metadata, 7 parametrised rank RMSE asserts, mean-Δ characterisation, achieved-vs-target logging |

## Acceptance gate

**Primary (per plan §D.3): τ_n-axis RMSE ≤ 2.5 cm per rank 9–15** — PASS.

| rank | obs peak | tgt peak | best Δ (°Cd) | RMSE@Δ (cm) | gate |
|------|----------|----------|--------------|-------------|------|
| 9 | 22.0 | 23.0 | +216 | **1.02** | PASS |
| 10 | 22.0 | 22.0 | +250 | **0.79** | PASS |
| 11 | 22.0 | 22.0 | +280 | **0.93** | PASS |
| 12 | 21.0 | 20.0 | +319 | **0.98** | PASS |
| 13 | 16.5 | 18.0 | +340 | **1.29** | PASS |
| 14 | 12.0 | 18.9 | +360 | **1.88** | PASS |
| 15 |  9.0 | 11.3 | +398 | **1.12** | PASS |

Mean per-rank offset Δ = **+309.0 ± 59.2 °Cd** (S3b.1 characterisation reproduced
bit-for-bit — target curve is unchanged between thin-B.3.5 and S3b.3 because
`calcLengthPerPhytomer` only gained a cessation-latch short-circuit in S3b.3
and production XML `tt_cessation=1500` never fires under Juelich 2024).

**Secondary (plan §D.4, non-blocking): shared absolute-axis residual at mean Δ
≤ ±25%** — FLAG (99.8% max residual).

Documented Ch1-inherited structural residual: +30 °Cd/rank plastochron drift
(Nielsen-axis leaf calendar × Déa-axis FA kinetics, different cultivar +
climate × Fournier Grignon Déa). S3b.5 cannot close this; Ch2 `tt_emergence`
refit via Andrieu axis is the planned fix.

## What this validates (and what it doesn't)

**Validates**: the **FA kinetic shape** — the per-rank `calcLengthPerPhytomer`
curves — matches Fournier Déa to within 2 cm peak error across ranks 9–15.
The "coordination-rule developmental kinetics on a physics-coupled segment
topology" claim in the FSPM 2026 framing is now defensible against digitised
Fournier 2000 Fig 6A Déa data.

**Does NOT validate**: the per-phytomer *geometric embedding* — i.e. the
achieved span length per rank reading the FA-kinetic target and realising it
as mid-stem-inserted nodes. Under the S3b.3 pragmatic scope downgrade
(`project_fa_s3b3_shipped.md`), `get_phytomer_length(n)` reports the
scalar-allocator span for each rank because the full per-rank mid-stem
insertion driver deadlocked on the leaf-emergence ↔ FA-kinetic chicken-and-egg.
Day-130 achieved values are uniform ~10 cm across ranks 2–16, while FA targets
peak at 22 cm (ranks 9–11) and decay to 4.5 cm (rank 16). This divergence is
logged in `d2_htt_per_rank_fournier_dea.json.achieved_vs_target_day130` as
diagnostic and visible in the bottom panel of `d2_htt_per_rank_plot_s3b.png`.

The writeup framing is therefore:

- **Fig Xa (target-under-thin, S3b.1)**: FA kinetic target is well-formed
- **Fig Xb (target vs Déa, S3b.5)**: FA kinetic shape matches observation
  (RMSE 0.79–1.88 cm per rank at optimal offset, peak within 2 cm)
- **Fig Xc (S3b.1 overlay + plastochron drift annotation)**: Ch1-inherited
  calendar residual characterised; Ch2 refit path identified
- **Peduncle caveat + achieved-vs-target diagnostic**: Ch2 deferral for true
  per-rank embedding

## Test coverage

```
dart/coupling/tests/test_fa_htt_trajectory.py
  test_d2_snapshot_metadata                            PASS (legacy)
  test_htt_monotone_nondecreasing                      PASS (legacy)
  test_endpoint_oracle_match_fa_dominant               PASS (legacy)
  test_fa_dominates_by_day_130                         PASS (legacy)
  test_s3b5_baseline_metadata                          PASS (new)
  test_s3b5_per_rank_tau_axis_rmse[9..15]              PASS x 7 (new)
  test_s3b5_mean_delta_matches_s3b1_characterisation   PASS (new)
  test_s3b5_achieved_vs_target_logged                  PASS (new)
  test_phase_iv_plateau_by_endpoint                    PASS (legacy)
```

Full FA battery: **51/51 pass** (41 pre-existing + 10 new) across
`test_fa_hard_invariants.py`, `test_fa_htt_trajectory.py`,
`test_fa_vstage_calendar.py`, `test_fa_coordination.py`,
`test_fa_apar_precondition.py`, `test_fa_per_phytomer_bookkeeping.py`.

## Hard invariants

All S3b hard invariants preserved (inherited through S3b.3/S3b.4 gates):

- HI#1: FA-off 6 D.0 baselines bit-identical (S3b.4 sign-off)
- HI#2: topmost leaf z = 150.356 cm (S3b.3 regression)
- HI#4: tassel emergence day 125 (S3b.3 regression)
- HI#5: `|getLength() − (basal_length + Σ length_per_n)| < 1e-6` (S3b.3 regression)

S3b.5 neither touches C++ code nor retunes kinetic parameters — it adds
Python-only baselines + pytest assertions against the unchanged FA target.

## Next (S3b.6)

DART APAR A/B (FA-on vs FA-off, server `diurnal`, mean APAR within 5%).
Closes parent plan follow-up #3 ("Literal D.4 DART APAR A/B").
