"""Phenology utilities — V-stage and tassel-stage detection from a CPlantBox plant.

Used by the production grow CLI to embed phenology labels into output filenames
under varying met forcing. Same collar-counting logic the lofter uses for
material assignment, so the label is consistent with the rendered geometry.

Labels returned by ``detect_v_stage``:
    V0..V18         vegetative stages (number of collared blades)
    VT_emerging     tassel present, not yet fully extended
    VT_mature       tassel fully extended, no significant senescence
    VT_senescent    tassel + visible basal-leaf senescence
"""
from __future__ import annotations

import math
from typing import Optional

import plantbox as pb

# Collar maturity thresholds — match cplantbox_adapter.py classification so
# labels agree with what the lofter actually renders as collared/emerging/whorl.
COLLAR_RELEASE = 0.45
COLLAR_THRESHOLD = 0.30

# Tassel extension fraction below which the tassel is still "emerging".
# 0.6 puts the boundary roughly at the silking transition under Juelich met.
TASSEL_EMERGING_FRAC = 0.6

# Average basal-leaf senescence ρ above which the canopy reads as senescent.
# 0.5 corresponds to ~R3 (mid-grain-fill) under the calibrated GDD onsets in
# cplantbox_adapter (positions 0/1/2 onset at 700/850/1000 °Cd, span 800 °Cd).
SENESCENT_RHO_THRESHOLD = 0.5


def count_visible_leaves(plant) -> dict:
    """Bucket every leaf by maturity = arc_length / lmax.

    Returns a dict ``{total, collared, emerging, whorl}`` matching the
    classification used by the NURBS adapter at render time.
    """
    n_total = n_coll = n_emrg = n_whrl = 0
    for organ in plant.getOrgans(pb.leaf):
        nodes = organ.getNodes()
        if len(nodes) < 2:
            continue
        lrp = organ.getLeafRandomParameter()
        lmax = max(float(lrp.lmax), 1e-9)
        cur = 0.0
        prev = nodes[0]
        for nd in nodes[1:]:
            dx = float(nd.x) - float(prev.x)
            dy = float(nd.y) - float(prev.y)
            dz = float(nd.z) - float(prev.z)
            cur += math.sqrt(dx * dx + dy * dy + dz * dz)
            prev = nd
        m = min(cur / lmax, 1.0)
        n_total += 1
        if m >= COLLAR_RELEASE:
            n_coll += 1
        elif m >= COLLAR_THRESHOLD:
            n_emrg += 1
        else:
            n_whrl += 1
    return {
        "total": n_total,
        "collared": n_coll,
        "emerging": n_emrg,
        "whorl": n_whrl,
    }


def is_v_stage_nielsen(counts: dict, target_v: int) -> bool:
    """Nielsen-compatible rule: V-n needs n collared AND ≥2 leaves above.

    Used by the time-series detector script to skip transient windows where
    leaf N has collared but leaf N+2 hasn't initiated yet. Not used by the
    one-shot ``detect_v_stage`` label below — that's a snapshot of the
    plant's current state, not a first-crossing detector.
    """
    return (counts["collared"] >= target_v
            and (counts["emerging"] + counts["whorl"]) >= 2)


def _tassel_extension_frac(plant) -> Optional[float]:
    """Max length / lmax across tassel organs (subType 20 spike, 21 branch).

    Returns ``None`` when no tassel organs exist yet.
    """
    best: Optional[float] = None
    for organ in plant.getOrgans(pb.stem):
        try:
            st = int(organ.getParameter("subType"))
        except Exception:
            continue
        if st not in (20, 21):
            continue
        srp = organ.getStemRandomParameter()
        lmax = max(float(srp.lmax), 1e-9)
        cur = float(organ.getLength())
        frac = min(cur / lmax, 1.0)
        if best is None or frac > best:
            best = frac
    return best


def _basal_senescence_rho(plant) -> float:
    """Average senescence progress ρ over basal leaf positions 0–2.

    Reuses ``cplantbox_adapter._senescence_progress`` so the threshold is
    consistent with the geometry-side senescence bend. Returns 0.0 when
    accumulated TT is unavailable (plant not yet stepped under met forcing).
    """
    from ..geometry.cplantbox_adapter import _senescence_progress

    if not hasattr(plant, "getAccumulatedTT"):
        return 0.0
    try:
        tt = float(plant.getAccumulatedTT())
    except Exception:
        return 0.0
    rhos = [_senescence_progress(p, tt, species="maize") for p in (0, 1, 2)]
    return sum(rhos) / len(rhos)


def detect_v_stage(plant) -> str:
    """One-shot phenology label for the plant's current state.

    Tassel takes precedence: any tassel organ → ``VT_*`` label. Otherwise
    the V-stage equals the number of collared blades. See module docstring
    for the full label set.
    """
    tassel_frac = _tassel_extension_frac(plant)
    if tassel_frac is not None:
        if _basal_senescence_rho(plant) >= SENESCENT_RHO_THRESHOLD:
            return "VT_senescent"
        if tassel_frac < TASSEL_EMERGING_FRAC:
            return "VT_emerging"
        return "VT_mature"

    n_coll = count_visible_leaves(plant)["collared"]
    return f"V{n_coll}"
