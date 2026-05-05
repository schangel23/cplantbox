"""Maize phloem transport parameters for CPlantBox PhloemFluxPython.

Literature-derived parameterization for Zea mays phloem sieve tubes,
following the structure of wheat_phloem_Giraud2023adapted.py but using
maize-specific measurements.

v2 (2026-02-25): Six simultaneous corrections based on primary literature
review (see Maize phloem transport model parameterization and validation.md):
  Q1: kx stem/leaf ratio 83x -> ~10x (anatomy + sieve plate corrections)
  Q2: Krm1 per-organ-type (leaf=0.25, stem=0.15, root=0.15) from WOFOST
  Q3: Bidirectional starch buffer (k_S_ST=1.0, kHyd_S_ST=1.0)
  Q4: Vmaxloading 0.05 -> 0.02 (reduced after other fixes)
  Q5: beta_loading 0.6 -> 2.0 (strong self-regulation)
  Q6: kr_st stem 0.0 -> 0.015 (transport phloem leakage-retrieval)
  +   C_targ 0.4 -> 0.8 (maize phloem 844-1400 mM, Ohshima 1990)

v3 (2026-05-05): Stem kx_st bug fix + V3 Babst 2022 falsification diagnostic.
  Audit chain (six PiafMunch sweeps + two anatomy audits, scripts in
  dart/coupling/scripts/pm_*) ran end-to-end against the Babst 2022 V3 maize
  measurement suite (T_chamber=20.75 C, DLI=30 mol/m2/d, 14:10 PP). Falsified
  five wrong hypotheses (385 mmol/d Q_Rmmax was a cumulative-integral misread;
  Krm1/Rho_s/StructSucrose unit chain clean; CSTimin lowering breaks V3 + wheat-55;
  Vmaxloading sweep saturates; Across_st audit clean) before isolating one bug:

  R1: Stem kx_st underspecified by 4.83x. The Babst k=0.91 um2 stem path bypassed
      the (N_bundles x numSE) anatomical multiplier that the file header
      documents (lines 99-110). Fix: apply full Hagen-Poiseuille per-SE x N_anatomy
      to stem only. Leaf left untouched (Babst's leaf k=0.23 um2 already
      averages across minor non-conducting veins; matches HP literal within 2x).
      JSON kx_st_stem: 5.86e-12 -> 2.83e-11 cm3 hPa-1 day-1.

  V3 outcome at corrected baseline (Vmaxloading=0.20 mmol cm-1 d-1 runtime override,
  CSTimin=0.20 unchanged, kx_st_stem corrected): 2/3 PASS vs Babst Table A1.
    [PASS] dP source-sink:   0.58 MPa (Babst 1.39 +/- 0.44, 2 sigma window)
    [PASS] C_ST_source:      0.45 mmol/cm3 (Babst 0.285 +/- 0.090, 2 sigma window)
    [FAIL] v sap basal:      0.14 m/hr (Babst 0.95 +/- 0.20, 2 sigma window)

  GATE checks all clean: wheat day-7.3 Lacointe regression ret=1, no C_ST blow-up;
  maize day-55 24h transient ret=1, Q_Rmmax=16 mmol/d stable, no regression.

  v shortfall is a methodological mismatch, not a parameter bug:
    - PiafMunch v probe = JW_ST / total_anatomical_Across_st at basal mainstem.
    - Babst v measurement = 11C tracer travel time / detector spacing at leaf
      base, sensitive only to actively-flowing SE at that moment.
    - Even with correct physics, the two probes report different observables:
      Jensen 2012 (line 41-43 below) gives effective transport area as 0.3-0.5x
      of anatomical sum; correcting for it symmetrically (both JW_ST and
      Across_st) cancels in v = JW_ST/Across_st. No clean single-knob fix exists.
    - Across_st audit confirmed clean (all ratios = 1.000 vs HP literal); v is
      structurally insensitive to all physiological parameters tested.
    - For absolute v matching: requires probe-method matching in PiafMunch output
      instrumentation (simulated tracer dispersion at leaf base), not parameter
      tuning. Deferred to Ch2 if needed; Ch1 transient-diagnostic use cases are
      insensitive to absolute v.

  Plant-size sanity check (V21d under Babst chamber proxy):
    mainstem 11 cm, 3 collared leaves, 8426 nodes, 548 root organs, 1690 cm root.
    Smaller than Babst V3 reference (30-50 cm, 4-6 leaves) -- L-mismatch is NOT
    the v shortfall explanation. Anomalous root:shoot ratio 150:1 flagged for
    separate investigation (likely XML calibration drift at young stages).

Key sources:
  Babst et al. (2022) Nature Plants 8:171-180
    — Sieve tube specific conductivity: k = 0.23 um2 (leaf), 0.91 um2 (stem)
    — Measured via SEM + Thompson & Holbrook (2003) formula
    — Includes sieve plate resistance (no need for separate beta correction)
    — Leaf k underrepresents export bundles (measured across all veins incl. minor)
  Russell & Evert (1985) Planta 164:448-458
    — Leaf vascular bundle counts: 30-50 total longitudinal bundles
    — Large + intermediate bundles have ~2 thin-walled SEs each
  Huang et al. (2016) J Integr Plant Biol
    — Stem bundles: 69-172 in uppermost internode (W22: ~110-130)
    — Not all bundles through-conducting (many are leaf traces)
    — Effective through-conducting: 60-80% of anatomical count
  Zhang et al. (2020) Plant Biotechnol J 19:35-50
    — Stem vascular bundles: 150-400+ (atactostele, genotype-dependent)
  Jansen et al. (2015) J Exp Bot 66:3949-3963
    — Root phloem poles: seminal 6-12, crown 15-30
  Evert et al. (1978) Planta 138:279-294
    — Sieve elements per bundle: 2-3 (small), 3-5 (intermediate), 5-10+ (large)
  Walsh & Evert (1975) Protoplasma 83:365-388
    — Stem SE diameter: 10-15 um
  Mullendore et al. (2010) Plant Cell 22:579
    — Phyllostachys nuda (bamboo, closest grass measured): SE radius 7.2 um
    — Maize export bundle SE radius estimated 4.0-5.0 um
  Jensen et al. (2012) Front Plant Sci 3:151
    — Sieve plate resistance ~ lumen resistance across 19 species
    — Correction factor 0.3-0.5 (Thompson & Holbrook 2003)
  Jensen et al. (2013) J R Soc Interface 10:20130055
    — Maize phloem 40.7% w/w (~1190 mM), among highest of 41 species
  Ohshima et al. (1990) Plant Cell Physiol 31:735
    — Maize phloem sap by stylectomy: 900-1400 mM
  Yesbergenova-Cuny et al. (2016) Plant Sci 252:347
    — Maize phloem by EDTA exudation: 844 mM
  Bihmidine et al. (2013, 2015) Front Plant Sci / BMC Plant Biol
    — Maize stems store primarily sucrose (not starch)
    — Sieve tubes symplastically isolated -> apoplastic unloading pathway
  Carpaneto et al. (2005) J Biol Chem 280:21437-21443
    — ZmSUT1: reversible, Km = 3.7 mM (apoplastic phloem loader)
  WOFOST (github.com/ajwdewit/WOFOST)
    — Maintenance respiration: leaf 0.030, stem 0.015, root 0.010-0.015 kg CH2O/kg DW/d
  Ben-Haj-Salah & Tardieu (1995) Plant Physiol 109:861-870
    — Leaf elongation rate: 48-72 mm/d (growth chamber, 24-26C)
  Tardieu et al. (2000) J Exp Bot 51:1505-1514
    — Leaf growth zone: 70-80 mm (temperature-invariant)
  Giaquinta (1983) Annu Rev Plant Physiol 34:347-387
    — Tissue-level Vmax loading: 2.3e-7 mol/m2/s
  Bloom et al. (2006) Ann Bot 97:867-873
    — Root elongation: seminal 36-53 mm/d
  Morrison et al. (2003) — Stem internode elongation: ~12 mm/d

Subtype mapping (from maize_calibrated.xml):
  Root: 1=taproot, 2=lateral1, 3=lateral2, 4=nodal, 5=shootborne
  Stem: 1=mainstem
  Leaf: 2=L0, 3=L1, 4=L2, 5=L3, 6=L4, 7=L5, 8=L6, 9=L7, 10=L8, 11=L9, 12=L10

PerType indexing: array[organType - 2][subType]
  organType 2 (root) -> index 0, 3 (stem) -> index 1, 4 (leaf) -> index 2
  subType used directly as array index (no offset)
"""

import numpy as np
import json
from pathlib import Path


# =====================================================================
# 1. VASCULAR ANATOMY
# =====================================================================

# Leaf blade widths from maize_calibrated.xml [cm]
# Used to scale vascular bundle count per leaf position
LEAF_WIDTHS = {
    2: 2.94, 3: 3.63, 4: 4.12, 5: 4.38, 6: 4.60,
    7: 4.42, 8: 4.29, 9: 4.13, 10: 3.81, 11: 3.63, 12: 3.20,
}
REF_WIDTH = 4.0  # cm — reference width for bundle scaling

# Number of vascular bundles
# Q1 FIX: Stem reduced from 250 to 175 effective through-conducting bundles
#   (Huang et al. 2016: 69-172 anatomical in upper internode; many are leaf traces)
# Q1 FIX: Leaf reduced from 40 to 35 export bundles (Russell & Evert 1985)
VascBundle_leaf_ref = 35   # Russell & Evert 1985 (large + intermediate export bundles)
VascBundle_stem = 175      # Huang et al. 2016 (effective through-conducting, ~70% of anatomical)
VascBundle_taproot = 20    # Crown/nodal: Jansen et al. 2015 (15-30)
VascBundle_lateral = 8     # Seminal/lateral: Jansen et al. 2015 (6-12)
VascBundle_nodal = 20      # Same as taproot (large adventitious)
VascBundle_shootborne = 15 # Intermediate (between lateral and crown)

# Sieve elements per bundle (thin-walled transport-active SEs only)
# Q1 FIX: Leaf reduced from 4 to 2 (Russell & Evert 1985: each large bundle has ~2 thin-walled SEs)
# Q1 FIX: Stem reduced from 4 to 3 (fewer effective transport SEs per bundle)
numSE_leaf = 2       # Russell & Evert 1985 (2 thin-walled SEs per export bundle)
numSE_stem = 3       # Esau 1943; Walsh & Evert 1975 (conservative: 3 of 3-6)
numSE_root_large = 3 # Esau 1943 (2-5 per phloem pole)
numSE_root_small = 2 # Smaller root types, fewer SE per pole

# Sieve element radii [cm]  (converted from um for CPlantBox CGS units)
# Q1 FIX: Leaf increased from 3.25 to 4.5 um — export bundles, not minor veins
#   (Mullendore et al. 2010: bamboo SE = 7.2 um; maize export SE estimated 4.0-5.0 um)
r_SE_leaf = 4.5e-4         # 4.5 um — export bundle SEs (Mullendore et al. 2010 extrapolation)
r_SE_stem = 6.25e-4        # 6.25 um — midpoint of 5.0-7.5 um (Walsh & Evert 1975)
r_SE_root_large = 5.5e-4   # 5.5 um  — midpoint of 4.0-7.5 um
r_SE_root_small = 4.5e-4   # 4.5 um  — lateral roots, narrower


# =====================================================================
# 2. AXIAL CONDUCTIVITY  (kx_st)  [cm^4, viscosity added by CPlantBox]
# =====================================================================
#
# Q1 FIX: The previous 83x stem/leaf kx ratio was biologically indefensible.
# Two key corrections:
#
# STEM: Hagen-Poiseuille with Thompson sieve plate correction and the same
#   effective bundle count from Q1 (175 through-conducting bundles).
#   kz_stem = N_bundles * N_SE * (r^4 * pi / 8) * beta_stem
#
# LEAF: Switch from Babst k to Hagen-Poiseuille with sieve plate correction.
#   Babst's leaf k=0.23 um^2 was measured across ALL veins including minor
#   ones, underrepresenting export bundle conductivity.  Export bundles have
#   larger SEs (r=4.5 um, not 3.25) and fewer but more effective elements.
#   kz_leaf = N_bundles * N_SE * (r^4 * pi / 8) * beta_plate
#
# ROOTS: Hagen-Poiseuille with beta=0.7 (unchanged).
#
# Target ratio: ~10x (was 83x).  Physiologically sensible: stem serves as
# shared conduit for 15-20 leaves (Jensen et al. 2012).

beta_stem = 0.9        # Thompson 2003a sieve plate correction
beta_plate = 0.5        # Jensen et al. 2012; Thompson & Holbrook 2003 (range 0.3-0.5)
beta_root = 0.7         # Jensen et al. (2012), angiosperm median


def _hp_kz(n_bundles, n_se, r_se, beta=None):
    """Hagen-Poiseuille kz [cm^4].  Viscosity added by CPlantBox."""
    if beta is None:
        beta = beta_root
    return n_bundles * n_se * (r_se ** 4) * np.pi / 8 * beta


# --- Per-leaf kz (HP with plate correction, scaled by blade width) ---
kz_per_leaf = {}
for _st, _w in LEAF_WIDTHS.items():
    _n_bun = VascBundle_leaf_ref * (_w / REF_WIDTH)
    kz_per_leaf[_st] = _hp_kz(_n_bun, numSE_leaf, r_SE_leaf, beta=beta_plate)

# --- Stem (HP with Thompson sieve plate correction) ---
kz_stem = _hp_kz(VascBundle_stem, numSE_stem, r_SE_stem, beta=beta_stem)

# --- Roots (Hagen-Poiseuille, no measured k) ---
kz_taproot = _hp_kz(VascBundle_taproot, numSE_root_large, r_SE_root_large)
kz_lateral1 = _hp_kz(VascBundle_lateral, numSE_root_small, r_SE_root_small)
kz_lateral2 = _hp_kz(VascBundle_lateral + 2, numSE_root_small, r_SE_root_small)
kz_nodal = _hp_kz(VascBundle_nodal, numSE_root_large, r_SE_root_large)
kz_shootborne = _hp_kz(VascBundle_shootborne, numSE_root_large, r_SE_root_large - 0.5e-4)


# =====================================================================
# 3. RADIAL CONDUCTIVITY  (kr_st)  [1/day]
# =====================================================================
#
# Leaves: 0 (source tissue — loading only, no unloading).
# Roots: symplastic unloading through plasmodesmata (terminal sink).
#   NO direct measurements for maize (see maize_parametarization.md, sec 3).
#   Use Giraud's wheat starting value (5e-2) as initial estimate.
#   THIS IS THE PRIMARY CALIBRATION PARAMETER — constrain against observed
#   carbon allocation patterns (~40% leaf / 30% stem / 30% root at day 55).
#
# Q6 FIX: Stems MUST have non-zero radial unloading conductivity.
#   kr_stem=0 made the stem a sealed pipe with no carbon exit path.
#   Maize stems actively unload sucrose via apoplastic pathway:
#     - Bihmidine et al. (2015): carboxyfluorescein shows symplastic isolation
#     - Patrick (1997): monocot stems use apoplastic component
#     - Setter & Meller (1984): 14C-sucrose uptake by maize stem tissues
#   Transport phloem has continuous leakage-retrieval (van Bel 2003).
#   Value: ~30% of root kr (intermediate sink, not terminal).

kr_leaf = 0.0
kr_stem = 0.015  # Q6: transport phloem leakage (Bihmidine 2015; ~30% of root value)
kr_root = 5e-2   # [1/d] — calibration parameter, start from Giraud wheat value

l_kr = 0.8  # cm — zone from root tip where exudation occurs


# =====================================================================
# 4. CROSS-SECTIONAL AREA  (Across_st)  [cm^2]
# =====================================================================

def _across(n_bundles, n_se, r_se):
    """Total sieve tube cross-sectional area [cm^2]."""
    return n_bundles * n_se * (r_se ** 2) * np.pi


# Note: Across uses the SAME updated anatomy constants (Q1 fix):
# leaf: 35 bundles × 2 SE × r=4.5um; stem: 175 bundles × 3 SE × r=6.25um
Across_per_leaf = {}
for _st, _w in LEAF_WIDTHS.items():
    _n_bun = VascBundle_leaf_ref * (_w / REF_WIDTH)
    Across_per_leaf[_st] = _across(_n_bun, numSE_leaf, r_SE_leaf)

Across_stem = _across(VascBundle_stem, numSE_stem, r_SE_stem)
Across_taproot = _across(VascBundle_taproot, numSE_root_large, r_SE_root_large)
Across_lateral1 = _across(VascBundle_lateral, numSE_root_small, r_SE_root_small)
Across_lateral2 = _across(VascBundle_lateral + 2, numSE_root_small, r_SE_root_small)
Across_nodal = _across(VascBundle_nodal, numSE_root_large, r_SE_root_large)
Across_shootborne = _across(VascBundle_shootborne, numSE_root_large, r_SE_root_large - 0.5e-4)


# =====================================================================
# 5. MAX GROWTH RATES  (Rmax_st)  [cm/d]
# =====================================================================
#
# Leaves: Ben-Haj-Salah & Tardieu (1995) — 48-72 mm/d (4.8-7.2 cm/d)
#   Position-dependent: mid-canopy leaves fastest (L4-L6), base/tip slower.
# Roots: Bloom et al. (2006) — 36-53 mm/d (3.6-5.3 cm/d) for seminal
#   Laterals slower (~15-20 mm/d).
# Stem: Morrison et al. (2003) — ~12 mm/d (1.2 cm/d) per internode

Rmax_per_leaf = {
    2: 4.8, 3: 5.5, 4: 6.0, 5: 6.5, 6: 7.0,
    7: 7.2, 8: 6.8, 9: 6.2, 10: 5.5, 11: 5.0, 12: 4.5,
}
Rmax_taproot = 4.5
Rmax_lateral1 = 1.5
Rmax_lateral2 = 2.0
Rmax_nodal = 4.5
Rmax_shootborne = 4.0
Rmax_stem = 1.2


# =====================================================================
# 6. SUCROSE DENSITY  (Rho_s)  [mmol Suc cm^-3]
# =====================================================================
# Tissue sucrose content per organ type (not per subtype).
# Maize stems store more sugar than wheat — slight increase.

Rho_s_root = 0.51   # same as wheat (Giraud)
Rho_s_stem = 0.70   # slightly higher than wheat (0.65) — maize stem sugar storage
Rho_s_leaf = 0.56   # same as wheat (Giraud)


# =====================================================================
# BUILD PerType ARRAYS
# =====================================================================
# Structure: [[root subtypes...], [stem subtypes...], [leaf subtypes...]]
#
# IMPORTANT: PiafMunch uses st2newst (MappedOrganism.cpp:mapSubTypes) to
# remap XML subtypes to 0-based sequential indices.  The PerType arrays
# MUST use 0-based indexing to match this remapping:
#   Root subtypes 1,2,3,4,5  -> remapped to 0,1,2,3,4
#   Stem subtype  1          -> remapped to 0
#   Leaf subtypes 2,3,...,12 -> remapped to 0,1,...,10
# NO zero-padding at index 0 — that would cause Across_ST=0 -> NaN.

def _build_root_array(taproot, lat1, lat2, nodal, shootborne):
    """Build root PerType array: 0-based (taproot=0, lat1=1, ...)."""
    return [taproot, lat1, lat2, nodal, shootborne]


def _build_stem_array(mainstem):
    """Build stem PerType array: 0-based (mainstem=0)."""
    return [mainstem]


def _build_leaf_array(per_leaf_dict):
    """Build leaf PerType array: 0-based (L2=0, L3=1, ..., L12=10)."""
    return [per_leaf_dict[st] for st in range(2, 13)]


def build_pertype():
    """Build all PerType arrays for JSON and setKr/setKx/setAcross calls."""
    return {
        "kx_st": [
            _build_root_array(kz_taproot, kz_lateral1, kz_lateral2, kz_nodal, kz_shootborne),
            _build_stem_array(kz_stem),
            _build_leaf_array(kz_per_leaf),
        ],
        "kr_st": [
            _build_root_array(kr_root, kr_root, kr_root, kr_root, kr_root),
            _build_stem_array(kr_stem),
            _build_leaf_array({st: kr_leaf for st in range(2, 13)}),
        ],
        "Across_st": [
            _build_root_array(Across_taproot, Across_lateral1, Across_lateral2,
                              Across_nodal, Across_shootborne),
            _build_stem_array(Across_stem),
            _build_leaf_array(Across_per_leaf),
        ],
        "Rmax_st": [
            _build_root_array(Rmax_taproot, Rmax_lateral1, Rmax_lateral2,
                              Rmax_nodal, Rmax_shootborne),
            _build_stem_array(Rmax_stem),
            _build_leaf_array(Rmax_per_leaf),
        ],
    }


# =====================================================================
# PUBLIC API
# =====================================================================

def setKrKx_phloem(r):
    """Set phloem conductivities on a PhloemFluxPython object.

    Follows Giraud et al. (2023) convention:
      r.setKr_st([[root subtypes], [stem subtypes], [leaf subtypes]])
      r.setKx_st(...)
      r.setAcross_st(...)

    Args:
        r: PhloemFluxPython (or compatible) object.

    Returns:
        The modified object (same reference).
    """
    pt = build_pertype()
    r.setKr_st(pt["kr_st"])
    r.setKx_st(pt["kx_st"])
    r.setAcross_st(pt["Across_st"])
    return r


def build_json():
    """Build the full phloem parameter dict for JSON serialization.

    Combines maize-specific PerType arrays with adapted global parameters.
    Based on phloem_parameters2025.json (wheat) with maize modifications.
    """
    pt = build_pertype()

    params = {
        "InitialValues": {
            "initValST": {"value": 0.4, "description": "Initial sucrose concentration in sieve tube [mmol/cm3]. Raised from 0.2 to start closer to C_targ=0.8"},
            "initValMeso": {"value": 0, "description": "Initial sucrose concentration in mesophyll"},
            "withInitVal": {"value": True, "description": "Use initial values"},
        },
        "Growth": {
            "psi_osmo_proto": {"value": -10078.8, "unit": "cm",
                               "description": "Osmotic potential in protophloem"},
            "psiMin": {"value": 2039.4, "unit": "cm",
                       "description": "Minimum water potential for growth"},
            "leafGrowthZone": {"value": 7.5, "unit": "cm",
                               "description": "Leaf growth zone length (Tardieu et al. 2000: 70-80 mm for maize, temperature-invariant)"},
            "Gr_Y": {"value": 0.8, "description": "Growth efficiency (g DW / g sucrose)"},
            "StemGrowthPerPhytomer": {"value": True, "description": "Growth per phytomer"},
            "useCWGr": {"value": True, "description": "Use carbon- and water-limited growth"},
        },
        "SieveTube": {
            "Vmaxloading": {"value": 0.02, "unit": "mmol cm-1 d-1",
                            "description": "Q4: Reduced from 0.05 after other fixes. "
                                           "Start low, increase if leaf export insufficient"},
            "CSTimin": {"value": 0.2,
                        "description": "Minimum sucrose concentration threshold for loading"},
            "beta_loading": {"value": 2.0,
                             "description": "Q5: Strong self-regulation (was 0.6). "
                                           "At C_ST=0.8: loading=20% of Vmax. "
                                           "Carpaneto 2005: ZmSUT1 inherently reversible"},
            "Mloading": {"value": 0.2,
                         "description": "Michaelis-Menten coefficient for loading [mmol Suc cm-3]. "
                                        "Note: molecular Km of ZmSUT1 = 3.7 mM (Carpaneto et al. 2005), "
                                        "but tissue-level apparent Km is higher"},
            "C_targ": {"value": 0.8, "unit": "mmol Suc cm-3",
                       "description": "Raised from 0.4 to match measured maize phloem concentration. "
                                      "Ohshima 1990: 900-1400 mM; Yesbergenova-Cuny 2016: 844 mM; "
                                      "Jensen 2013: 40.7% w/w (~1190 mM)"},
            "Q10": {"value": 2.0,
                    "description": "Q10 for maintenance respiration (Di Matteo 2018)"},
            "TrefQ10": {"value": 20.0, "unit": "\u00b0C",
                        "description": "Reference temperature for Q10"},
            "KMfu": {"value": 0.11,
                     "description": "Michaelis-Menten coefficient for sucrose usage"},
            "k_mucil": {"value": 0.0, "unit": "d-1",
                        "description": "Mucilage decay rate"},
            "k_mucil_": {"value": [], "description": "Vector of mucilage decay rates"},
            "Vmax_S_ST": {"value": 0.0, "unit": "mmol Suc d-1 cm-3",
                          "description": "Max sucrose usage in sieve tube"},
            "kM_S_ST": {"value": 0.0, "unit": "mmol Suc cm-3",
                        "description": "Michaelis-Menten constant for ST sucrose usage"},
            "kHyd_S_ST": {"value": 1.0, "unit": "d-1",
                          "description": "Q3: Starch hydrolysis (was 0.0 = irreversible trap). "
                                         "Must be >0 for bidirectional starch buffer. "
                                         "Stems remobilize reserves during grain fill. "
                                         "kHyd >= k_S -> sucrose dominates at equilibrium"},
            "k_S_ST": {"value": 1.0, "unit": "d-1",
                       "description": "Q3: Starch synthesis (was 20.0 = absurdly fast, t1/2~50 min). "
                                      "Stem starch is minor, slow-turnover pool. "
                                      "Bihmidine 2013: maize stems store primarily sucrose, not starch"},
            "update_viscosity_": {"value": True,
                                  "description": "Update viscosity with concentration (eta ~ exp(4.68*c))"},
            "usePsiXyl": {"value": True,
                          "description": "Couple sieve tube water potential with xylem potential"},
        },
        "Mesophyll": {
            "C_targMesophyll": {"value": 0.8, "unit": "mmol Suc cm-3",
                                "description": "Matches C_targ (raised from 0.4)"},
            "Vmax_S_Mesophyll": {"value": 0.0, "unit": "mmol Suc d-1 cm-3",
                                 "description": "Max sucrose usage in mesophyll"},
            "kM_S_Mesophyll": {"value": 0.0, "unit": "mmol Suc cm-3",
                               "description": "Michaelis-Menten constant for mesophyll"},
            "kHyd_S_Mesophyll": {"value": 0.0, "unit": "d-1",
                                 "description": "Sucrose hydrolysis rate in mesophyll"},
            "k_S_Mesophyll": {"value": 1.0, "unit": "d-1",
                              "description": "Matches k_S_ST (was 20.0)"},
            "surfMeso": {"value": 0.01, "unit": "cm2",
                         "description": "Cross-sectional area of mesophyll"},
            "sameVolume_meso_seg": {"value": True,
                                    "description": "Same volume for mesophyll and segment"},
            "sameVolume_meso_st": {"value": False,
                                   "description": "Same volume for sieve tube and mesophyll"},
        },
        "PerType": {
            "Across_st": {
                "value": pt["Across_st"],
                "unit": "cm2",
                "description": "Total sieve tube cross-sectional area per organ subtype",
            },
            "kr_st": {
                "value": pt["kr_st"],
                "unit": "1 d-1",
                "description": "Q6: Phloem radial conductivity. Leaf=0 (source). "
                               "Stem=0.015 (transport phloem leakage, Bihmidine 2015). "
                               "Root=0.05 (terminal sink, CALIBRATION PARAMETER).",
            },
            "kx_st": {
                "value": pt["kx_st"],
                "unit": "cm4",
                "description": "Q1: Stem from Babst (2022) k with reduced bundles (175). "
                               "Leaf from HP with plate correction (beta=0.5). "
                               "Roots from HP with beta=0.7. Stem/leaf ratio ~10x (was 83x).",
            },
            "Krm2": {"value": [[3e-05], [2e-05], [4e-05]],
                     "description": "Q2: Per-organ-type (root, stem, leaf). "
                                    "Proportionally scaled with Krm1"},
            "Krm1": {"value": [[0.15], [0.15], [0.25]],
                     "description": "Q2: Per-organ-type maintenance respiration (WOFOST). "
                                    "Leaf=0.25 (2.5x; protein-rich), stem=0.15, root=0.15. "
                                    "Total respiration: ~9 -> ~18 mmol/d (50-65% of loading)"},
            "Rho_s": {
                "value": [[Rho_s_root], [Rho_s_stem], [Rho_s_leaf]],
                "unit": "mmol Suc cm-3",
                "description": "Sucrose density per organ type (root, stem, leaf)",
            },
            "Rmax_st": {
                "value": pt["Rmax_st"],
                "unit": "cm d-1",
                "description": "Maximum growth rate per subtype. Leaf: Ben-Haj-Salah & Tardieu 1995. "
                               "Root: Bloom et al. 2006. Stem: Morrison et al. 2003.",
            },
        },
        "Soil": {
            "DefaultC": {"value": 0.0001, "unit": "mmol Suc cm-3",
                         "description": "Dummy value for soil sucrose concentration"},
        },
        "Solver": {
            "atol": {"value": 1e-4, "description": "Base absolute tolerance (scaled per variable type internally)"},
            "rtol": {"value": 1e-4, "description": "Relative tolerance"},
            "doTroubleshooting": {"value": False, "description": "Enable troubleshooting output"},
        },
    }
    return params


def print_summary():
    """Print a diagnostic summary of computed phloem parameters."""
    pt = build_pertype()
    print("=" * 70)
    print("Maize Phloem Parameters — Diagnostic Summary")
    print("=" * 70)
    print("NOTE: PerType arrays use 0-based indexing (st2newst remapping)")

    print("\n--- Axial conductivity kx_st [cm^4] ---")
    labels_r = ["taproot", "lateral1", "lateral2", "nodal", "shootborne"]
    for i, (val, lab) in enumerate(zip(pt["kx_st"][0], labels_r)):
        print(f"  Root  idx={i} ({lab:12s}): {val:.4e}")
    print(f"  Stem  idx=0 (mainstem)  : {pt['kx_st'][1][0]:.4e}")
    for i in range(len(pt["kx_st"][2])):
        xml_st = i + 2  # leaf subtypes start at 2 in XML
        print(f"  Leaf  idx={i:2d} (L{i}, W={LEAF_WIDTHS.get(xml_st, 0):.2f} cm): "
              f"{pt['kx_st'][2][i]:.4e}")

    # Compute and display kx ratio
    mid_leaf_st = 6  # L4, widest leaf
    mid_leaf_idx = mid_leaf_st - 2
    kx_leaf_mid = pt['kx_st'][2][mid_leaf_idx]
    kx_stem_val = pt['kx_st'][1][0]
    print(f"\n  ** Stem/Leaf kx ratio (L4): {kx_stem_val / kx_leaf_mid:.1f}x **")

    print("\n--- Radial conductivity kr_st [1/d] ---")
    print(f"  Roots: {kr_root}  (CALIBRATION PARAMETER)")
    print(f"  Stem:  {kr_stem}  (Q6: transport phloem leakage)")
    print(f"  Leaf:  {kr_leaf}")

    print("\n--- Cross-sectional area Across_st [cm^2] ---")
    print(f"  Stem (mainstem): {Across_stem:.6e}")
    print(f"  Leaf (L4, max):  {Across_per_leaf[6]:.6e}")
    print(f"  Root (taproot):  {Across_taproot:.6e}")

    print("\n--- Max growth rate Rmax_st [cm/d] ---")
    print(f"  Leaf range: {min(Rmax_per_leaf.values()):.1f} - {max(Rmax_per_leaf.values()):.1f}")
    print(f"  Stem:       {Rmax_stem}")
    print(f"  Taproot:    {Rmax_taproot}")

    print("\n--- 0-based indexing (st2newst remapping) ---")
    print(f"  Root: XML st [1,2,3,4,5] -> idx [0,1,2,3,4]  (5 entries)")
    print(f"  Stem: XML st [1]         -> idx [0]            (1 entry)")
    print(f"  Leaf: XML st [2..12]     -> idx [0..10]        (11 entries)")


# =====================================================================
# CLI: generate JSON when run as standalone script
# =====================================================================

if __name__ == "__main__":
    import sys

    print_summary()

    out_dir = Path(__file__).parent
    out_path = out_dir / "phloem_parameters_maize2026.json"

    params = build_json()
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"\nJSON written to: {out_path}")

    # Also copy to coupling data dir if it exists
    coupling_data = Path(__file__).resolve().parents[3] / "dart" / "coupling" / "data"
    if coupling_data.is_dir():
        copy_path = coupling_data / "phloem_parameters_maize2026.json"
        with open(copy_path, "w") as f:
            json.dump(params, f, indent=2)
        print(f"JSON copied to: {copy_path}")
