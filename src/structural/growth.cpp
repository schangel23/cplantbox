// MultiPhaseStemGrowth — implementation
//
// Bodies live here (not in growth.h) because the implementation needs
// the full definitions of Stem, StemRandomParameter, StemSpecificParameter,
// Leaf, and Plant — pulling those into growth.h would create a circular
// include with Stem.h → Organ.h → growth.h.
//
// Numerical correspondence: this class is the GF-side port of the FA
// length-calc that previously lived as an `if (p.use_fournier_andrieu_kinetics)`
// shadow branch in Stem::simulate (deleted in S0.5, 2026-04-28). The outer
// cessation-latch block (Stem.cpp ~158-260) still runs in Stem::simulate
// and writes to Stem mirror fields; those are sync'd to/from this GF's
// per_organ_state around the f_gf->getLength call. Pure-scalar contract
// per Lock #4 of ADR_LEAF_KINEMATICS_2026-04-28: no geometry side effects
// (createSegments / createLateral / span-walk stay in Stem::simulate
// after the dispatch).
//
// S0.5 status: every FA-on stem dispatches through this class via the
// native f_gf chain. Full retirement of Stem-side mirror fields (move
// length_per_n / cessation_age_ / etc. fully onto per_organ_state, retire
// Stem::syncStateFromGeometry, expose GF state via Pybind so tests stop
// reading Stem mirrors) is deferred to S0.5b.

#include "growth.h"

#include <algorithm>
#include <cmath>

#include "Leaf.h"
#include "Plant.h"
#include "Stem.h"
#include "leafparameter.h"
#include "stemparameter.h"

namespace CPlantBox {

// FA kinetic constants. Mirrors the anonymous namespace in Stem.cpp:1086-1089.
// Defined locally so the GF impl is self-contained.
namespace {
constexpr double IL_INIT_CM = 0.0025;            // Zhu 2014: initial IL at tau=0
constexpr double IL_AT_END_PHASE_II_CM = 4.5;    // FA 2000 line 223, phyt 7-15
constexpr double HALF_PLASTOCHRON_LAG_DEGCD = 9.6;  // FA 2000 line 207
}

// -------------------------------------------------------------------------
// Helper: lazy-allocate per_organ_state[organId] sized from the LRP.
// Idempotent — repeated calls are no-ops once sized.
// -------------------------------------------------------------------------
MultiPhaseStemGrowth::PerOrganFAState&
MultiPhaseStemGrowth::ensureState(int organId, std::shared_ptr<const Organ> o) const
{
    auto& st = per_organ_state[organId];
    if (!o) return st;

    auto stem = std::static_pointer_cast<const Stem>(o);
    auto srp = stem->getStemRandomParameter();
    if (!srp) return st;
    auto sp = std::static_pointer_cast<const StemSpecificParameter>(stem->param());
    if (!sp) return st;

    const int n_ranks = static_cast<int>(sp->internode_v_n.size());
    const int n_laterals_max = static_cast<int>(sp->ln.size()) + 1;
    const int per_n_end = std::max(n_ranks + 1, n_laterals_max + 1);

    if (static_cast<int>(st.length_per_n.size()) < per_n_end) {
        st.length_per_n.resize(per_n_end, 0.0);
        st.epsilonDx_per_n.resize(per_n_end, 0.0);
        st.cessation_age_per_n.resize(per_n_end, -1.0);
        st.cessation_andrieu_tt_per_n.resize(per_n_end, -1.0);
        st.lateral_spawned_per_n.resize(per_n_end, 0);
    }
    if (static_cast<int>(st.initiation_andrieu_tt_per_n.size()) < per_n_end) {
        st.initiation_andrieu_tt_per_n.resize(per_n_end, -1.0);
    }
    return st;
}

// -------------------------------------------------------------------------
// Per-rank target length (FA Phase I→IV). Pure formula; no state mutation.
// Mirrors Stem::calcLengthPerPhytomer (Stem.cpp:1108-1222) with τ_n anchor
// resolution preferring this GF's initiation_andrieu_tt_per_n[n] over
// leaf-emergence fallback (S3b.7 plastochron path).
// -------------------------------------------------------------------------
double MultiPhaseStemGrowth::calcLengthPerPhytomer(int n,
                                                    std::shared_ptr<const Organ> o) const
{
    if (!o) return 0.0;
    auto stem = std::static_pointer_cast<const Stem>(o);
    auto srp = stem->getStemRandomParameter();
    if (!srp) return 0.0;
    auto sp = std::static_pointer_cast<const StemSpecificParameter>(stem->param());
    if (!sp) return 0.0;

    // Basal-zero ranks (Zhu 2014 line 127): zero-length at all τ.
    const auto& basal_zero = srp->basal_zero_ranks;
    if (std::find(basal_zero.begin(), basal_zero.end(), n) != basal_zero.end()) {
        return 0.0;
    }

    // τ_n anchor resolution: prefer plastochron-driven init_tt when set
    // (S3b.7), else fall back to leaf emergence + half-plastochron lag.
    auto state_it = per_organ_state.find(o->getId());
    double init_tt = -1.0;
    if (state_it != per_organ_state.end()
        && n >= 1
        && n < static_cast<int>(state_it->second.initiation_andrieu_tt_per_n.size())
        && state_it->second.initiation_andrieu_tt_per_n[n] >= 0.0) {
        init_tt = state_it->second.initiation_andrieu_tt_per_n[n];
    } else {
        // Walk children looking for the n-th leaf (1-based ordinal).
        // const_cast around getNumberOfChildren / getChild because Organ
        // declares them non-const (legacy signature); the calls don't
        // mutate state, only iterate. Const-correctness fix is upstream.
        auto stem_mut = std::const_pointer_cast<Stem>(stem);
        int leaf_ordinal = 0;
        double leaf_emerge_tt = -1.0;
        const int n_children = stem_mut->getNumberOfChildren();
        for (int ci = 0; ci < n_children; ++ci) {
            auto c = stem_mut->getChild(ci);
            if (c->organType() == Organism::ot_leaf) {
                ++leaf_ordinal;
                if (leaf_ordinal == n) {
                    auto lf = std::static_pointer_cast<Leaf>(c);
                    leaf_emerge_tt = lf->getEmergenceAndrieuTT();
                    break;
                }
            }
        }
        if (leaf_emerge_tt < 0.0) return 0.0;  // not yet emerged
        init_tt = leaf_emerge_tt + HALF_PLASTOCHRON_LAG_DEGCD;
    }

    auto plant_fa = stem->getPlant();
    if (!plant_fa) return 0.0;
    double andrieu_tt = plant_fa->getAccumulatedAndrieuTT();

    // Cessation freeze: per-rank latch dominates over global latch when set
    // (matches Stem::calcLengthPerPhytomer:1167-1173). Per-rank latches live
    // on the GF state; the global latch lives on Stem (cessation_andrieu_tt_)
    // because use_thermal_cessation is not FA-specific.
    if (state_it != per_organ_state.end()) {
        const auto& st = state_it->second;
        if (n >= 1
            && n < static_cast<int>(st.cessation_andrieu_tt_per_n.size())
            && st.cessation_andrieu_tt_per_n[n] >= 0.0
            && andrieu_tt > st.cessation_andrieu_tt_per_n[n]) {
            andrieu_tt = st.cessation_andrieu_tt_per_n[n];
        } else if (stem->cessation_andrieu_tt_ >= 0.0
                   && andrieu_tt > stem->cessation_andrieu_tt_) {
            andrieu_tt = stem->cessation_andrieu_tt_;
        }
    } else if (stem->cessation_andrieu_tt_ >= 0.0
               && andrieu_tt > stem->cessation_andrieu_tt_) {
        andrieu_tt = stem->cessation_andrieu_tt_;
    }
    const double tau = andrieu_tt - init_tt;
    if (tau < 0.0) return 0.0;

    const double r_I = srp->r_I;
    const double phase_I_duration = srp->phase_I_duration;
    const double phase_II_duration = srp->phase_II_duration;
    const double phase_IV_duration = srp->phase_IV_duration;
    const double phase_IV_k = srp->phase_IV_k;

    // Phase I: pre-collar exponential.
    if (tau < phase_I_duration) {
        return IL_INIT_CM * std::exp(r_I * tau);
    }
    // Phase II: linear ramp from end-of-Phase-I to 4.5 cm uniform boundary.
    const double phase_II_end = phase_I_duration + phase_II_duration;
    if (tau < phase_II_end) {
        const double IL_end_I = IL_INIT_CM * std::exp(r_I * phase_I_duration);
        const double frac = (tau - phase_I_duration) / phase_II_duration;
        return IL_end_I + frac * (IL_AT_END_PHASE_II_CM - IL_end_I);
    }
    // Phase III: per-rank linear at v_n for D_n.
    const auto& vvec = sp->internode_v_n;
    const auto& dvec = sp->internode_D_n;
    if (n < 1 || n > static_cast<int>(vvec.size()) || n > static_cast<int>(dvec.size())) {
        return IL_AT_END_PHASE_II_CM;
    }
    const double v_n = vvec[n - 1];
    const double D_n = dvec[n - 1];
    const double phase_III_end = phase_II_end + D_n;
    if (tau < phase_III_end) {
        return IL_AT_END_PHASE_II_CM + v_n * (tau - phase_II_end);
    }
    // Phase IV: exponential decay toward IL_final.
    const double IL_end_III = IL_AT_END_PHASE_II_CM + v_n * D_n;
    const auto& ilfvec = sp->internode_IL_final;
    if (n < 1 || n > static_cast<int>(ilfvec.size())) {
        return IL_end_III;
    }
    const double IL_final = ilfvec[n - 1];
    return IL_final - (IL_final - IL_end_III) * std::exp(-phase_IV_k * (tau - phase_III_end));
}

// -------------------------------------------------------------------------
// Sum of per-rank target lengths.
// Geometry-side readers call this through the (organId, o) pair.
// -------------------------------------------------------------------------
double MultiPhaseStemGrowth::calcLengthPerPhytomerSum(int organId,
                                                      std::shared_ptr<const Organ> o) const
{
    if (!o) return 0.0;
    auto stem = std::static_pointer_cast<const Stem>(o);
    auto sp = std::static_pointer_cast<const StemSpecificParameter>(stem->param());
    if (!sp) return 0.0;
    const int n_ranks = static_cast<int>(sp->internode_v_n.size());
    double total = 0.0;
    for (int n = 1; n <= n_ranks; ++n) {
        total += calcLengthPerPhytomer(n, o);
    }
    return total;
}

// -------------------------------------------------------------------------
// Per-rank latched length accessor.
// Mirrors today's Stem::getPhytomerLength (Stem.cpp:1288-1294).
// -------------------------------------------------------------------------
double MultiPhaseStemGrowth::getPhytomerLength(int organId, int n) const
{
    auto it = per_organ_state.find(organId);
    if (it == per_organ_state.end()) return 0.0;
    const auto& lpn = it->second.length_per_n;
    if (n < 1 || n >= static_cast<int>(lpn.size())) return 0.0;
    return lpn[n];
}

// -------------------------------------------------------------------------
// Cessation latch update (idempotent).
// Mirrors the outer per-rank cessation block in Stem.cpp:158-260.
// Walks leaf children, computes τ_n per rank, latches when threshold
// crossed. Fires global cessation_age once all per-rank latches are set.
//
// Threshold dispatch (S0.6 / Lock #1 — three-way):
//   delayNGEndAxis == TT && delayNGEnd > 0  → merged Andrieu-TT crossing
//                                              (Lock #1 form, replaces a
//                                              tt_cessation-shaped sibling)
//   tt_cessation > 0                        → legacy global plant-TT crossing
//   else                                    → per-rank Phase IV operational
//                                              completion (phase_I + phase_II
//                                              + D_n[n] + phase_IV_dur)
// -------------------------------------------------------------------------
static void update_cessation_latches(MultiPhaseStemGrowth::PerOrganFAState& st,
                                     std::shared_ptr<const Organ> o,
                                     std::shared_ptr<const StemRandomParameter> srp,
                                     double age)
{
    if (!srp->use_thermal_cessation) return;
    auto stem = std::static_pointer_cast<const Stem>(o);
    auto plant_cess = stem->getPlant();
    if (!plant_cess) return;

    const int n_ranks = static_cast<int>(srp->internode_v_n.size());
    if (n_ranks <= 0) return;

    // S0.6 / Lock #1: merged-form axis-TT path takes precedence over legacy
    // tt_cessation; both are global plant-TT thresholds (constant per rank).
    // When neither applies, the per-rank Phase IV operational completion path
    // is used. Existing XMLs default delayNGEndAxis=Calendar so axis_tt_active
    // is false for every pre-S0.6 file → bit-identical regression preserved.
    const bool axis_tt_active = srp->delayNGEndAxis == DelayAxis::TT
                                && srp->delayNGEnd > 0.0;
    const bool legacy_threshold = srp->tt_cessation > 0.0;
    const double plant_andrieu_tt = plant_cess->getAccumulatedAndrieuTT();

    // const_cast for child iteration (same reason as in calcLengthPerPhytomer).
    auto stem_mut = std::const_pointer_cast<Stem>(stem);
    int leaf_ordinal = 0;
    const int n_children = stem_mut->getNumberOfChildren();
    for (int ci = 0; ci < n_children; ++ci) {
        auto c = stem_mut->getChild(ci);
        if (c->organType() != Organism::ot_leaf) continue;
        ++leaf_ordinal;
        if (leaf_ordinal > n_ranks) break;
        if (leaf_ordinal >= static_cast<int>(st.cessation_andrieu_tt_per_n.size())) break;
        if (st.cessation_andrieu_tt_per_n[leaf_ordinal] >= 0.0) continue;

        // Prefer plastochron-driven init_tt (S3b.7) over leaf emergence.
        double init_tt = -1.0;
        if (leaf_ordinal < static_cast<int>(st.initiation_andrieu_tt_per_n.size())
            && st.initiation_andrieu_tt_per_n[leaf_ordinal] >= 0.0) {
            init_tt = st.initiation_andrieu_tt_per_n[leaf_ordinal];
        } else {
            auto lf = std::static_pointer_cast<Leaf>(c);
            const double leaf_tt = lf->getEmergenceAndrieuTT();
            if (leaf_tt < 0.0) continue;
            init_tt = leaf_tt + HALF_PLASTOCHRON_LAG_DEGCD;
        }
        const double tau_n = plant_andrieu_tt - init_tt;
        double threshold;
        if (axis_tt_active) {
            // Lock #1 merged form: delayNGEnd is the Andrieu-TT global cessation
            // threshold (interpreted in °Cd via the axis flag).
            threshold = srp->delayNGEnd;
        } else if (legacy_threshold) {
            threshold = srp->tt_cessation;
        } else {
            const std::size_t d_idx = static_cast<std::size_t>(leaf_ordinal - 1);
            const double D_n = (d_idx < srp->internode_D_n.size())
                ? srp->internode_D_n[d_idx]
                : 0.0;
            threshold = srp->phase_I_duration
                      + srp->phase_II_duration
                      + D_n
                      + srp->phase_IV_duration;
        }
        if (tau_n >= threshold) {
            st.cessation_andrieu_tt_per_n[leaf_ordinal] = plant_andrieu_tt;
            st.cessation_age_per_n[leaf_ordinal] = age;
        }
    }

    // S0.5b.5: the global all-latched gate now lives on Stem (cessation_age_
    // / cessation_andrieu_tt_ are non-FA-specific use_thermal_cessation
    // feature state). The outer block in Stem::simulate fires the gate
    // unconditionally each step using these very same per-rank latches; no
    // need to duplicate it here.
}

// -------------------------------------------------------------------------
// Plastochron forecast (idempotent).
// Mirrors the lazy plastochron block in Stem.cpp:420-451.
// For each non-spawned rank whose plastochron birthday has crossed,
// seeds length_per_n[n] = basal_internode_cm so fa_sum can include the
// to-be-created segment's budget.
// -------------------------------------------------------------------------
static void update_plastochron_forecast(MultiPhaseStemGrowth::PerOrganFAState& st,
                                        std::shared_ptr<const Organ> o,
                                        std::shared_ptr<const StemRandomParameter> srp,
                                        std::shared_ptr<const StemSpecificParameter> sp)
{
    auto stem = std::static_pointer_cast<const Stem>(o);
    auto plant_fc = stem->getPlant();
    const double plant_andrieu_tt = plant_fc ? plant_fc->getAccumulatedAndrieuTT() : 0.0;
    const double plastochron = srp->plastochron_andrieu;
    const double basal_step = srp->basal_internode_cm;
    const int n_laterals_max = static_cast<int>(sp->ln.size()) + 1;

    for (int n = 1; n <= n_laterals_max; ++n) {
        if (n >= static_cast<int>(st.lateral_spawned_per_n.size())) break;
        if (st.lateral_spawned_per_n[n]) continue;
        const double init_tt_n = static_cast<double>(n) * plastochron;
        if (plant_andrieu_tt < init_tt_n) break;  // ascending gate
        // Topmost lateral attaches without an internode (matches the scalar-
        // burst "extra createLateral outside the for-loop" semantics).
        if (n >= n_laterals_max) continue;
        if (n >= static_cast<int>(st.length_per_n.size())) break;
        if (st.length_per_n[n] < basal_step) {
            st.length_per_n[n] = basal_step;
        }
    }
}

// -------------------------------------------------------------------------
// MultiPhaseStemGrowth::getLength
// Idempotent FA target length computation. Returns the analytical
// p.lb + min(fa_sum, Σ p.ln) — caller adds epsilonDx afterward.
// Cessation-latched: returns the organ's current realised length so
// e = targetlength - length = 0 in the caller.
//
// Defensive fallback: if the SP's FA flag is off (caller assigned this
// GF to a non-FA stem), fall through to ExponentialGrowth's formula.
// The factory only mints this GF for FA-on stems, so the fallback
// should never fire in normal dispatch.
// -------------------------------------------------------------------------
double MultiPhaseStemGrowth::getLength(double t, double r, double k,
                                        std::shared_ptr<Organ> o) const
{
    if (!o) return 0.0;
    auto stem = std::static_pointer_cast<Stem>(o);
    auto srp = stem->getStemRandomParameter();
    auto sp = std::static_pointer_cast<const StemSpecificParameter>(stem->param());
    if (!srp || !sp) return 0.0;

    if (!sp->use_fournier_andrieu_kinetics) {
        return k * (1.0 - std::exp(-(r / k) * t));
    }

    auto& st = ensureState(o->getId(), o);

    // Block 1: cessation latch update (per-rank + global).
    update_cessation_latches(st, o, srp, /*age=*/t);

    // Block 2: plastochron forecast (seed length_per_n for crossed ranks).
    update_plastochron_forecast(st, o, srp, sp);

    // Block 3: fa_sum across n_ranks (use max of FA target vs. allocated).
    const int n_ranks = static_cast<int>(sp->internode_v_n.size());
    double fa_sum = 0.0;
    for (int n = 1; n <= n_ranks; ++n) {
        const double target_n = calcLengthPerPhytomer(n, o);
        const double allocated_n = (n < static_cast<int>(st.length_per_n.size()))
                                    ? st.length_per_n[n]
                                    : 0.0;
        const double driver_n = (target_n > allocated_n) ? target_n : allocated_n;
        fa_sum += driver_n;
    }

    // Block 4: branching cap. min(fa_sum, sum(ln)) — matches today's clamp.
    double ln_sum = 0.0;
    for (double v : sp->ln) ln_sum += v;
    double targetlength = sp->lb + ((fa_sum < ln_sum) ? fa_sum : ln_sum);

    // Block 5: cessation gate. Force dl=0 in the caller by returning current
    // realised length (so e = targetlength - length = 0).
    // S0.5b.5: the global cessation latch is canonical on Stem
    // (use_thermal_cessation feature is not FA-specific). Read from there.
    if (stem->cessation_age_ >= 0.0) {
        return o->getLength(true);
    }
    return targetlength;
}

// -------------------------------------------------------------------------
// MultiPhaseStemGrowth::getAge
// Closed-form piecewise inverse. Null-safe for Plant::initCallbacks's
// gf->getAge(1,1,1,nullptr) probe (Lock #2 of ADR_LEAF_KINEMATICS_2026-04-28).
//
// Limitation: the FA length law on a stem is the SUM of per-rank
// piecewise functions, not a single per-rank curve. There is no
// analytical "age such that total stem length = L" inverse for the
// multi-rank case (each rank contributes asynchronously). The honest
// answer is to return ExponentialGrowth's inverse as a defensive
// fallback (matches the "scalar else-branch" that today's
// Stem::calcAge dispatches to via f_gf when the FA flag is off).
//
// In practice getAge is only invoked from lateral-creation bookkeeping
// (Organ.cpp:888, 914-916) and from Stem::calcAge — neither is on the
// per-step elongation hot path, so the fallback is acceptable until
// S0.5 / S2 prove a real consumer needs the multi-rank inverse.
// -------------------------------------------------------------------------
double MultiPhaseStemGrowth::getAge(double l, double r, double k,
                                     std::shared_ptr<const Organ> o) const
{
    if (!o) return 0.0;  // initCallbacks probe — Lock #2 null guard
    if (l <= 0) return 0.0;
    if (l >= k) return 1.e9;
    const double age = -k / r * std::log(1.0 - l / k);
    return std::isfinite(age) ? age : 1.e9;
}

// -------------------------------------------------------------------------
// MultiPhaseStemGrowth::syncStateFromGeometry
// Recomputes length_per_n[n] from realised segment lengths after
// Stem::simulate's createSegments / createLateral fire. Mirrors today's
// post-hoc span-walk in Stem.cpp:706-744.
//
// In relative coordinates (hasRelCoord()==true) nodes[k] for k>=1 IS
// the segment delta vector from node k-1 to k.
// -------------------------------------------------------------------------
void MultiPhaseStemGrowth::syncStateFromGeometry(std::shared_ptr<const Organ> o,
                                                  const std::vector<int>& node_to_phytomer,
                                                  double basal_length) const
{
    if (!o) return;
    auto it = per_organ_state.find(o->getId());
    if (it == per_organ_state.end()) return;  // never seen by getLength
    auto& st = it->second;
    st.basal_length = basal_length;

    const auto& nodes = o->getNodes();
    const int n_nodes = static_cast<int>(nodes.size());

    // Extend length_per_n to cover all rank tags that appear in
    // node_to_phytomer (apical/peduncle rank n_links is one past the
    // topmost rank — must be in range to close Hard Invariant #5).
    int max_tag = 0;
    for (int k = 0; k < n_nodes && k < static_cast<int>(node_to_phytomer.size()); ++k) {
        if (node_to_phytomer[k] > max_tag) max_tag = node_to_phytomer[k];
    }
    if (static_cast<int>(st.length_per_n.size()) < max_tag + 1) {
        st.length_per_n.resize(max_tag + 1, 0.0);
        st.epsilonDx_per_n.resize(max_tag + 1, 0.0);
        st.cessation_age_per_n.resize(max_tag + 1, -1.0);
        st.cessation_andrieu_tt_per_n.resize(max_tag + 1, -1.0);
        st.lateral_spawned_per_n.resize(max_tag + 1, 0);
    }

    const int n_ranks_lpn = static_cast<int>(st.length_per_n.size()) - 1;
    if (n_ranks_lpn < 1) return;
    for (int n = 0; n <= n_ranks_lpn; ++n) {
        st.length_per_n[n] = 0.0;
    }
    for (int k = 1; k < n_nodes && k < static_cast<int>(node_to_phytomer.size()); ++k) {
        const int tag = node_to_phytomer[k];
        if (tag < 1 || tag > n_ranks_lpn) continue;
        st.length_per_n[tag] += nodes[k].length();
    }
}


// =========================================================================
// MultiPhaseLeafGrowth — Andrieu/Hillier/Birch 2006 piecewise leaf kinetics
// (cv. Déa hybrid lineage with MF3D L_fin endpoint anchor; per-rank
// scalars on LeafRandomParameter, see ADR_LEAF_KINEMATICS_2026-04-28 §D1).
//
// Length law (C¹ at the exp/lin junction):
//   Phase E (exp):   t ∈ [T0, T1)   → L = L_min · exp(R1 · (t − T0))
//   Phase L (lin):   t ∈ [T1, T2)   → L = L1 + R2 · (t − T1)
//   Plateau:         t ≥ T2         → L = L_fin
// with T1 = T0 + lag_exp, T2 = T1 + D_lin,
//      L1 = L_min · exp(R1 · lag_exp),
//      L_fin = (coordinated_lmax > 0) ? coordinated_lmax : getK()  (Lock #5).
//
// t is the plant's Andrieu-axis TT (Tb=9.8 °C — Plant::andrieu_tt_,
// accessed via Plant::getAccumulatedAndrieuTT). The legacy
// accumulatedTT_ (Tb=8) is untouched by this class — Lock #5 of the
// ADR retires the original D3 "both submodels read the same axis"
// reading because doing so silently rebinds the legacy axis that
// non-Andrieu leaf XMLs depend on.
//
// Scope (Lock #4): pure scalar target length. Geometry side effects
// (createSegments, branching-zone bookkeeping) stay in Leaf::simulate
// after the dispatch. The class holds no per-organ state — every
// dispatch reads kinetics fresh from o->getLeafRandomParameter(),
// and no bookkeeping needs to persist across calls (a leaf is one
// rank, not many phytomers like a stem).
// =========================================================================
double MultiPhaseLeafGrowth::getLength(double t, double r, double k,
                                        std::shared_ptr<Organ> o) const
{
    if (!o) return 0.0;
    auto leaf = std::static_pointer_cast<Leaf>(o);
    auto lrp = leaf->getLeafRandomParameter();
    if (!lrp) return 0.0;

    // Empty-array silent freeze guard (Lock #6 minor finding 10): when
    // R1_n is unconfigured, fall through to ExponentialGrowth so a
    // misconfigured XML produces a visible scalar curve rather than a
    // frozen zero-length leaf. The factory only mints this GF for
    // opted-in subTypes (gf=6 in XML), so the fallback is defensive.
    if (lrp->R1_n <= 0.0) {
        return k * (1.0 - std::exp(-(r / k) * t));
    }

    auto plant = o->getPlant();
    if (!plant) {
        // No plant attached — return zero so caller's `e = target - length`
        // collapses to a no-op. Defensive; in normal dispatch the plant
        // is always set by the time getLength fires.
        return 0.0;
    }
    const double tt = plant->getAccumulatedAndrieuTT();

    // Reads canopy-coordinated lmax when set (M9 / Lock #5), else the
    // realised getK(). The caller already passed `k = param()->getK()`,
    // so we override only when sibling-leaf coordination is active.
    const double k_final = (leaf->coordinated_lmax > 0.0) ? leaf->coordinated_lmax : k;

    const double R1 = lrp->R1_n;
    const double R2 = lrp->R2_n;
    const double T0 = lrp->T0_n;
    const double T1 = T0 + lrp->lag_exp_n;
    const double T2 = T1 + lrp->D_lin_n;
    const double L_min = (lrp->L_min > 0.0) ? lrp->L_min : 0.025;
    const double L1 = L_min * std::exp(R1 * lrp->lag_exp_n);

    double L;
    if (tt < T0) {
        // Pre-initiation: leaf has not entered Phase E. Andrieu's L_min
        // is the "tip emergence" length; below T0 we still return L_min
        // so existing-but-not-yet-elongating leaves carry a sensible
        // microscopic length rather than zero (zero would let the caller
        // wipe length to zero on every step).
        L = L_min;
    } else if (tt < T1) {
        L = L_min * std::exp(R1 * (tt - T0));
    } else if (tt < T2) {
        L = L1 + R2 * (tt - T1);
    } else {
        // Plateau — saturate at L_fin. L_fin honours MF3D / coordinated_lmax.
        L = L1 + R2 * lrp->D_lin_n;
    }

    // Honour the canopy-coordination cap unconditionally — even on the
    // plateau, sibling coordination may shrink k_final below the Andrieu
    // L_fin (rare under Chapter 1 maize calibration but needed for
    // wheat / sorghum / shaded canopies).
    if (L > k_final) L = k_final;
    return L;
}

// -------------------------------------------------------------------------
// MultiPhaseLeafGrowth::getAge — closed-form piecewise inverse, null-safe
// for Plant::initCallbacks's gf->getAge(1,1,1,nullptr) probe (Lock #2).
// -------------------------------------------------------------------------
double MultiPhaseLeafGrowth::getAge(double l, double r, double k,
                                     std::shared_ptr<const Organ> o) const
{
    if (!o) return 0.0;  // initCallbacks probe — Lock #2 null guard
    auto leaf = std::static_pointer_cast<const Leaf>(o);
    auto lrp = leaf->getLeafRandomParameter();
    if (!lrp || lrp->R1_n <= 0.0) {
        // Unconfigured: fall back to ExponentialGrowth's analytical inverse.
        if (l <= 0) return 0.0;
        if (l >= k) return 1.e9;
        const double age = -k / r * std::log(1.0 - l / k);
        return std::isfinite(age) ? age : 1.e9;
    }

    const double R1 = lrp->R1_n;
    const double R2 = lrp->R2_n;
    const double T0 = lrp->T0_n;
    const double T1 = T0 + lrp->lag_exp_n;
    const double T2 = T1 + lrp->D_lin_n;
    const double L_min = (lrp->L_min > 0.0) ? lrp->L_min : 0.025;
    const double L1 = L_min * std::exp(R1 * lrp->lag_exp_n);
    const double L_fin = L1 + R2 * lrp->D_lin_n;

    if (l < L_min) return T0;
    if (l < L1) {
        return T0 + std::log(l / L_min) / R1;
    }
    if (l < L_fin) {
        return (R2 > 0.0) ? (T1 + (l - L1) / R2) : T2;
    }
    return T2;  // saturated
}

} // namespace CPlantBox
