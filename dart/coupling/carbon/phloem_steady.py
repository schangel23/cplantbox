"""Quasi-steady-state phloem transport solver on a vascular tree.

Eliminates the ODE stiffness problem in PiafMunch by setting dC_ST/dt = 0
at every node, then solving the steady-state balance algebraically via
tree-sweep Picard iteration. Physically justified because Münch pressure
flow equilibrates in seconds-minutes while carbon allocation changes over
hours.

The equations match PiafMunch2.cpp:153-230 exactly:
  Loading:   Q_Fl = Vmax * L_blade * C_meso/(Km + C_meso) * exp(-C_ST * beta)
  Usage:     Fu = (Q_Rmmax + Q_Grmax) * C_ST/(C_ST + KMfu)
  Exudation: Q_Exud = (C_ST - C_soil) * Q_Exudmax_coeff

For sugar transport conductance (Münch flow), the effective conductance
per segment is: K_sugar = kx * R*T * C_avg / (mu * L) where R*T provides
the osmotic-to-pressure conversion and mu is the sucrose-dependent viscosity.

Algorithm: O(N) tree-sweep Picard iteration (no linear algebra needed).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tree_topology import VascularTree, build_tree


@dataclass
class PhloemParams:
    """Phloem transport parameters loaded from JSON."""
    Vmaxloading: float      # mmol cm-1 d-1
    beta_loading: float     # dimensionless
    C_targ: float           # mmol Suc cm-3
    Mloading: float         # mmol Suc cm-3 (Michaelis-Menten for loading)
    CSTimin: float          # mmol Suc cm-3 (minimum for usage)
    Q10: float
    TrefQ10: float          # C
    KMfu: float             # mmol Suc cm-3 (Michaelis-Menten for usage)
    Gr_Y: float             # growth efficiency
    leafGrowthZone: float   # cm
    C_soil: float           # mmol Suc cm-3
    k_S_ST: float           # d-1, starch synthesis rate (sucrose → starch)
    kHyd_S_ST: float        # d-1, starch hydrolysis rate (starch → sucrose)

    # PerType arrays: [root_subtypes, stem_subtypes, leaf_subtypes]
    kx_st: list
    kr_st: list
    Across_st: list
    Rmax_st: list
    Krm1: list              # per organ type (not per subtype)
    Krm2: list              # per organ type
    Rho_s: list             # per organ type


def load_phloem_params(species="maize") -> PhloemParams:
    """Load phloem parameters from the coupling data JSON.

    Args:
        species: Species name (looks for phloem_parameters_{species}2026.json).

    Returns:
        PhloemParams with all values extracted from JSON.
    """
    from ..config import DATA_DIR

    json_path = DATA_DIR / f"phloem_parameters_{species}2026.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Phloem params not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    st = data["SieveTube"]
    pt = data["PerType"]

    return PhloemParams(
        Vmaxloading=st["Vmaxloading"]["value"],
        beta_loading=st["beta_loading"]["value"],
        C_targ=st["C_targ"]["value"],
        Mloading=st["Mloading"]["value"],
        CSTimin=st["CSTimin"]["value"],
        Q10=st["Q10"]["value"],
        TrefQ10=st["TrefQ10"]["value"],
        KMfu=st["KMfu"]["value"],
        Gr_Y=data["Growth"]["Gr_Y"]["value"],
        leafGrowthZone=data["Growth"]["leafGrowthZone"]["value"],
        C_soil=data["Soil"]["DefaultC"]["value"],
        k_S_ST=st["k_S_ST"]["value"],
        kHyd_S_ST=st["kHyd_S_ST"]["value"],
        kx_st=pt["kx_st"]["value"],
        kr_st=pt["kr_st"]["value"],
        Across_st=pt["Across_st"]["value"],
        Rmax_st=pt["Rmax_st"]["value"],
        Krm1=pt["Krm1"]["value"],
        Krm2=pt["Krm2"]["value"],
        Rho_s=pt["Rho_s"]["value"],
    )


def _pertype_lookup(array, organ_type, sub_type):
    """Look up a PerType value using the st2newst remapping convention.

    PerType indexing (from maize_phloem_2026.py):
      Root (ot=2): XML subType 1,2,3,4,5  -> JSON index 0,1,2,3,4
      Stem (ot=3): XML subType 1          -> JSON index 0
      Leaf (ot=4): XML subType 2,3,...,12  -> JSON index 0,1,...,10

    If the array has only 1 entry per organ type (Krm1, Krm2, Rho_s),
    return that single value regardless of subtype.
    """
    ot_idx = organ_type - 2  # 2->0, 3->1, 4->2
    if ot_idx < 0 or ot_idx >= len(array):
        return 0.0

    organ_array = array[ot_idx]
    if len(organ_array) == 0:
        return 0.0

    # Single-value arrays (Krm1, Krm2, Rho_s) — same for all subtypes
    if len(organ_array) == 1:
        return organ_array[0]

    # Multi-value: remap subtype to 0-based index
    if organ_type == 2:    # root: st 1-5 -> idx 0-4
        idx = sub_type - 1
    elif organ_type == 3:  # stem: st 1 -> idx 0
        idx = sub_type - 1
    elif organ_type == 4:  # leaf: st 2-12 -> idx 0-10
        idx = sub_type - 2
    else:
        return 0.0

    idx = max(0, min(idx, len(organ_array) - 1))
    return organ_array[idx]


def _sucrose_viscosity_mPas(C_mmol_cm3, T_C=25.0):
    """Sucrose solution viscosity [mPa·s] (Mathlouthi & Genotelle 1995).

    Matches PiafMunch's runPM.cpp viscosity computation.
    """
    # Density (water-based, simplified)
    d_water = 997.0  # kg/m3 at 25C (simplified)
    # Sucrose mass concentration: C [mmol/cm3] * 342.3 [g/mol] / 1000 [mmol/mol] = g/cm3
    c_g_cm3 = C_mmol_cm3 * 342.3 / 1000.0
    d = c_g_cm3 * 1000 + d_water  # density in g/L (approximate)

    # Sucrose content (g sucrose / g solution) as fraction
    sc = (c_g_cm3 * 1000) / d * 100.0  # percent
    si_phi = (30.0 - T_C) / (91.0 + T_C)
    si_n = sc / (1900.0 - 18.0 * sc) if (1900.0 - 18.0 * sc) > 0 else 0.0

    if si_n <= 0:
        return 0.89  # pure water at 25C

    log_mu = (22.46 * si_n) - 0.114 + si_phi * (1.1 + 43.1 * si_n**1.25)
    # Clamp to prevent overflow
    log_mu = min(log_mu, 10.0)
    return 10.0 ** log_mu


def _sugar_transport_conductance(kx, L, C_avg, T_C=25.0):
    """Effective sugar transport conductance [mmol/d per (mmol/cm3)].

    In Münch flow, osmotic pressure drives bulk flow:
      Q_vol = kx/(mu*L) * dP,  with dP = R*T*dC (van't Hoff)
      J_suc = Q_vol * C_avg = kx*R*T*C_avg/(mu*L) * dC

    Args:
        kx: Axial conductivity [cm^4] (includes all sieve elements).
        L: Segment length [cm].
        C_avg: Average sucrose concentration [mmol/cm3].
        T_C: Temperature [C].

    Returns:
        K_sugar [mmol/d per (mmol/cm3)] — sugar flux per concentration difference.
    """
    if kx <= 0 or L <= 0:
        return 0.0

    mu_mPas = _sucrose_viscosity_mPas(C_avg, T_C)
    # Convert mPa·s to hPa·d: divide by (1e5 mPa/hPa * 86400 s/d) = 8.64e9
    mu_hPa_d = mu_mPas / 8.64e9

    # R*T in hPa·cm3/mmol
    R = 8.314  # J/(mol·K) = Pa·m3/(mol·K)
    T_K = T_C + 273.15
    # R*T [Pa·m3/mol] = R*T [Pa·(1e6 cm3)/mol] = R*T*1e6 [Pa·cm3/mol]
    # Convert Pa to hPa: /100. Then mol to mmol: /1000
    RT_hPa_cm3_mmol = R * T_K * 1e6 / 100.0 / 1000.0  # = R*T * 10.0

    K = kx * RT_hPa_cm3_mmol * C_avg / (mu_hPa_d * L)
    return K


class QuasiSteadyPhloem:
    """Quasi-steady-state phloem transport solver.

    Solves for steady-state sucrose concentration C_ST at each node by
    iterating a tree-sweep (Picard) algorithm. No ODE integration needed.
    """

    # Conversion: 1 mol CO2 -> mmol Suc
    # Photosynthesis: 6 CO2 -> 1 glucose, 2 glucose -> 1 sucrose
    # So 12 CO2 -> 1 sucrose.  1 mol CO2 * 1000/12 = 83.33 mmol Suc
    CO2_TO_SUC = 1000.0 / 12.0

    # Inverse: 1 mmol Suc respired -> 12 mmol CO2 released.
    # Used to convert Rm/Rg outputs to mmol CO2/d (matching DVS solver interface).
    SUC_TO_CO2 = 12.0

    def __init__(self, plant, tree=None, params=None, sim_day=None):
        """Initialize the solver from a CPlantBox plant.

        Args:
            plant: pb.MappedPlant (grown, with soil grid).
            tree: Pre-built VascularTree (optional, built from plant if None).
            params: PhloemParams (optional, loaded from JSON if None).
            sim_day: Simulation day for DVS-dependent sink modulation.
                If provided, root and leaf sink strengths are attenuated
                based on WOFOST developmental tables (FRTB, FLTB).
        """
        self.plant = plant
        self.tree = tree if tree is not None else build_tree(plant)
        self.params = params if params is not None else load_phloem_params()
        self.sim_day = sim_day
        self._dvs_sink_factors = self._compute_dvs_factors(sim_day)
        self._precompute_node_params()

    def _compute_dvs_factors(self, sim_day):
        """Compute DVS-dependent sink attenuation factors.

        Real maize progressively shuts down root allocation as it approaches
        flowering (DVS=1.0) and shifts carbon to stems and storage organs.
        WOFOST's FRTB encodes this developmental program. We use the ratio
        FRTB(DVS) / FRTB(0) to scale root sink conductivities, and similarly
        FLTB(DVS) / FLTB(0) for leaf growth after DVS ~0.88.

        This is biologically justified: root apices produce fewer laterals,
        meristematic activity declines, and assimilate demand shifts to
        elongating internodes and eventually ear development.

        Note: the ratio is applied uniformly to all root segments' Rmax and
        kr. With ~1700 root segments vs ~60 stem segments, this still leaves
        roots collectively capturing more growth carbon than WOFOST's FRTB
        target at high DVS. The remaining gap (~50% vs 80% stem at DVS=0.83)
        is a known architectural constraint of the segment-based approach.
        A future refinement could implement per-meristem quiescence (binary
        on/off per root tip) rather than uniform rate reduction.

        Returns:
            dict with 'root' and 'leaf' attenuation factors in [0, 1].
        """
        if sim_day is None:
            return {'root': 1.0, 'leaf': 1.0}

        from .dvs_partitioning import _dvs_from_day, _interp_table, FRTB, FLTB

        dvs = _dvs_from_day(sim_day)

        # Root attenuation: FRTB(DVS) / FRTB(0)
        fr_root_now = _interp_table(FRTB, dvs)
        fr_root_max = _interp_table(FRTB, 0.0)  # 0.40 at emergence
        root_factor = fr_root_now / fr_root_max if fr_root_max > 0 else 0.0

        # Leaf attenuation: FLTB(DVS) / FLTB(0)
        # FLTB declines from 0.62 to 0.15 by DVS=0.88, then to 0 by DVS=1.2
        fr_leaf_now = _interp_table(FLTB, dvs)
        fr_leaf_max = _interp_table(FLTB, 0.0)  # 0.62 at emergence
        leaf_factor = fr_leaf_now / fr_leaf_max if fr_leaf_max > 0 else 0.0

        print(f"  DVS sink modulation: DVS={dvs:.2f}, "
              f"root_factor={root_factor:.3f}, leaf_factor={leaf_factor:.3f}")

        return {'root': root_factor, 'leaf': leaf_factor}

    def _precompute_node_params(self):
        """Precompute per-node parameters from tree topology and phloem params."""
        t = self.tree
        p = self.params
        N = t.n_nodes

        # Per-node arrays
        self.vol_ST = np.zeros(N)          # sieve tube volume [cm3]
        self.Q_Rmmax_base = np.zeros(N)    # base maintenance resp rate [mmol/d]
        self.krm2_node = np.zeros(N)       # concentration-dependent Rm coefficient
        self.loading_len = np.zeros(N)     # blade length for loading [cm]
        self.K_sugar = np.zeros(N)         # sugar transport conductance [mmol/d per mmol/cm3]
        self.Q_Exudmax_coeff = np.zeros(N) # exudation coefficient
        self.Q_Grmax_node = np.zeros(N)    # max growth rate [mmol Suc/d]
        self.storage_vol = np.zeros(N)     # stem parenchyma volume for starch storage [cm3]
        self.is_source = np.zeros(N, dtype=bool)

        segments = self.plant.getSegments()

        for seg_idx in range(t.n_segments):
            seg = segments[seg_idx]
            child_node = seg.y
            ot = int(t.organ_type[seg_idx])
            st = int(t.sub_type[seg_idx])
            L = t.seg_length[seg_idx]

            if L <= 0:
                continue

            # Sieve tube cross-section and volume
            Across = _pertype_lookup(p.Across_st, ot, st)
            self.vol_ST[child_node] = Across * L

            # Sugar transport conductance (Münch flow with osmotic conversion)
            kx = _pertype_lookup(p.kx_st, ot, st)
            self.K_sugar[child_node] = _sugar_transport_conductance(
                kx, L, p.C_targ, T_C=25.0
            )

            # Maintenance respiration: Q_Rmmax = krm1 * rho_s * vol_seg
            krm1 = _pertype_lookup(p.Krm1, ot, st)
            rho_s = _pertype_lookup(p.Rho_s, ot, st)
            struct_suc = rho_s * t.seg_vol[seg_idx]
            self.Q_Rmmax_base[child_node] = krm1 * struct_suc
            self.krm2_node[child_node] = _pertype_lookup(p.Krm2, ot, st)

            # Loading: only on leaf blade segments
            blade_len = t.blade_length[seg_idx]
            if blade_len > 0:
                self.loading_len[child_node] = blade_len
                self.is_source[child_node] = True

            # Exudation: only on below-ground root segments
            # DVS attenuation: root exudation declines as plant approaches
            # flowering (fewer active root tips, less radial transport).
            if t.is_root_below[seg_idx]:
                kr = _pertype_lookup(p.kr_st, ot, st)
                kr *= self._dvs_sink_factors['root']
                radius = t.seg_radius[seg_idx]
                self.Q_Exudmax_coeff[child_node] = 2 * np.pi * radius * L * kr

            # Stem storage: parenchyma volume for starch accumulation.
            # CPlantBox stem radius (a=0.2 cm) represents vascular bundle
            # dimensions, NOT the full internode. Real maize internodes are
            # 1.5–2.5 cm diameter (Morrison et al. 1994). The storage pool
            # is pith + cortex parenchyma (~90% of internode cross-section).
            # Use a realistic internode radius for storage volume.
            if ot == 3:
                STEM_PARENCHYMA_RADIUS = 0.9  # cm (mature maize, conservative)
                parenchyma_vol = np.pi * STEM_PARENCHYMA_RADIUS**2 * L
                self.storage_vol[child_node] = parenchyma_vol

        # Growth demand: computed from organ-level maturity, NOT sieve tube
        # cross-section. Uses the actual organ tissue cross-section and only
        # assigns demand to segments within the growth zone of growing organs.
        self._compute_growth_demand()

    def _compute_growth_demand(self):
        """Compute per-node growth demand from organ maturity and growth zones.

        PiafMunch computes growth from CPlantBox's internal deltaSucOrgNode
        (actual volume growth per timestep). We approximate this by:
        1. Identifying growing organs (length < 95% of lmax)
        2. Computing growth demand from organ tissue cross-section x Rmax
        3. Applying only to segments within the growth zone

        Growth zones:
        - Leaf: basal leafGrowthZone cm (meristematic zone)
        - Stem: per-internode intercalary meristems. Real maize stem
          elongation occurs simultaneously at 3-5 phytomers via
          intercalary meristematic activity, NOT at a single apical
          meristem. We identify internode boundaries from leaf attachment
          nodes and assign a growth zone within each immature internode.
        - Root: tip segment only (~0.2 cm)
        """
        t = self.tree
        p = self.params
        segments = self.plant.getSegments()

        # Build (parent_node, child_node) -> global segment index lookup
        seg_lookup = {}
        for si in range(t.n_segments):
            seg = segments[si]
            seg_lookup[(seg.x, seg.y)] = si

        ROOT_GROWTH_ZONE = 0.1   # tip elongation zone (0.5-1mm meristem)

        # Pre-compute stem internode growth zones (shared across all stem organs)
        stem_growing_si = self._compute_stem_internode_zones(seg_lookup)

        for org in self.plant.getOrgans(-1):
            ot = org.organType()
            if ot < 2:
                continue

            try:
                lmax = org.getParameter("lmax")
                r = org.getParameter("r")
            except Exception:
                continue

            if lmax <= 0 or r <= 0:
                continue
            if org.getLength() >= lmax * 0.95:
                continue  # fully grown

            st = org.getParameter("subType")
            Rmax = _pertype_lookup(p.Rmax_st, ot, int(st))
            rho_s = _pertype_lookup(p.Rho_s, ot, int(st))
            if Rmax <= 0 or rho_s <= 0:
                continue

            # DVS-dependent growth attenuation: root and leaf meristematic
            # activity declines as the plant transitions to reproductive growth.
            if ot == 2:   # root
                Rmax *= self._dvs_sink_factors['root']
            elif ot == 4:  # leaf
                Rmax *= self._dvs_sink_factors['leaf']

            # Get this organ's global segment indices (ordered base -> tip)
            org_segs = org.getSegments()
            org_si = []
            for s in org_segs:
                key = (s.x, s.y)
                if key in seg_lookup:
                    org_si.append(seg_lookup[key])

            if not org_si:
                continue

            # Determine growth zone (indices within organ)
            if ot == 4:  # leaf: basal growth zone
                gz_cm = p.leafGrowthZone
                cum = 0.0
                gz_count = 0
                for si in org_si:
                    cum += t.seg_length[si]
                    gz_count += 1
                    if cum >= gz_cm:
                        break
                growing_si = org_si[:gz_count]
            elif ot == 3:  # stem: per-internode intercalary growth zones
                growing_si = stem_growing_si
            elif ot == 2:  # root: tip
                gz_cm = ROOT_GROWTH_ZONE
                cum = 0.0
                gz_count = 0
                for si in reversed(org_si):
                    cum += t.seg_length[si]
                    gz_count += 1
                    if cum >= gz_cm:
                        break
                growing_si = org_si[-gz_count:]
            else:
                continue

            # Assign growth demand to segments in the growth zone
            for si in growing_si:
                L = t.seg_length[si]
                if L <= 0:
                    continue
                organ_cross = t.seg_vol[si] / L  # tissue cross-section [cm2]
                Q_gr = Rmax * organ_cross * rho_s / p.Gr_Y
                child_node = segments[si].y
                self.Q_Grmax_node[child_node] = Q_gr

    def _compute_stem_internode_zones(self, seg_lookup):
        """Identify per-internode growth zones on the main stem.

        Maize stem elongation at DVS 0.5-0.9 occurs via intercalary
        meristems at the base of each internode, with 3-5 phytomers
        actively elongating simultaneously (Morrison et al. 1994).

        We find leaf attachment nodes to define internode boundaries,
        then assign a growth zone (basal 3 cm per internode) to each
        internode whose stem is still growing (< 95% of lmax).

        Returns:
            list of global segment indices in growth zones.
        """
        import plantbox as pb

        t = self.tree
        segments = self.plant.getSegments()

        # Find the main stem organ
        stems = [o for o in self.plant.getOrgans()
                 if o.organType() == pb.OrganTypes.stem]
        if not stems:
            return []

        main_stem = stems[0]
        stem_lmax = main_stem.getParameter("lmax")
        stem_length = main_stem.getLength(False)
        if stem_length >= stem_lmax * 0.95:
            return []  # fully grown

        # Get stem segment indices
        stem_segs = main_stem.getSegments()
        stem_si = []
        for s in stem_segs:
            key = (s.x, s.y)
            if key in seg_lookup:
                stem_si.append(seg_lookup[key])

        if not stem_si:
            return []

        # Get all node positions and find leaf attachment z-heights.
        # Leaf attachment = first node of each leaf organ = shared stem node.
        all_nodes = self.plant.getNodes()
        leaves = [o for o in self.plant.getOrgans()
                  if o.organType() == pb.OrganTypes.leaf]
        attach_z = sorted(set(
            o.getNodes()[0].z for o in leaves if len(o.getNodes()) > 0
        ))

        if len(attach_z) < 2:
            # Fallback: single growth zone at stem tip
            return stem_si[-50:]  # ~5 cm at dx=0.1

        # Build segment z-midpoint lookup
        seg_z = {}
        for si in stem_si:
            seg = segments[si]
            n0 = all_nodes[seg.x]
            n1 = all_nodes[seg.y]
            seg_z[si] = (n0.z + n1.z) / 2.0

        # Per-internode growth zone: basal 3 cm of each internode.
        # Intercalary meristems sit at the base of elongating internodes
        # (just above the node/leaf insertion), NOT at the tip.
        #
        # Maize internode growth follows an acropetal wave: lower internodes
        # finish elongating first, upper ones are still active. We check
        # each internode's current length against its expected mature length
        # (= total stem lmax / n_internodes) to determine if it's still growing.
        INTERNODE_GZ_CM = 3.0   # intercalary meristem zone per internode
        n_internodes = len(attach_z) - 1
        # Expected mature internode length (uniform for simplicity;
        # real maize has graduated internodes but this is a good average).
        mature_internode = stem_lmax / max(n_internodes, 1)

        growing_si = []
        n_active = 0
        for i in range(n_internodes):
            z_base = attach_z[i]
            z_top = attach_z[i + 1] if i + 1 < len(attach_z) else z_base + mature_internode
            current_len = z_top - z_base

            # Skip internodes that have reached mature length
            if current_len >= mature_internode * 0.95:
                continue

            n_active += 1
            # Growth zone: basal INTERNODE_GZ_CM of this internode
            gz_top = z_base + INTERNODE_GZ_CM
            for si in stem_si:
                z = seg_z.get(si, -999)
                if z_base <= z < min(gz_top, z_top):
                    growing_si.append(si)

        # Also include the apical zone above the last leaf (new internodes forming)
        if attach_z:
            apical_base = attach_z[-1]
            stem_tip_z = max(seg_z.values()) if seg_z else apical_base
            if stem_tip_z > apical_base + 1.0:
                n_active += 1
                gz_top = apical_base + INTERNODE_GZ_CM
                for si in stem_si:
                    z = seg_z.get(si, -999)
                    if apical_base <= z <= stem_tip_z:
                        growing_si.append(si)

        if n_active > 0:
            print(f"  Stem internode growth zones: {n_active} active "
                  f"of {n_internodes} internodes, {len(growing_si)} segments")

        return growing_si

    def _map_An_to_nodes(self, An_per_leaf_seg):
        """Map leaf-segment net assimilation to mesophyll sucrose source per node.

        CPlantBox's get_net_assimilation() returns one value per leaf segment,
        ordered by their position in getSegments() filtered to organType==4.
        We convert mol CO2/d -> mmol Suc/d and assign to corresponding nodes.

        Args:
            An_per_leaf_seg: np.array of shape (n_leaf_segs,), mol CO2/d.

        Returns:
            np.array of shape (n_nodes,), mmol Suc/d per node.
        """
        t = self.tree
        An_nodes = np.zeros(t.n_nodes)

        # Find leaf segment indices (matching CPlantBox ordering)
        leaf_seg_indices = np.where(t.organ_type == 4)[0]
        assert len(leaf_seg_indices) == len(An_per_leaf_seg), (
            f"Leaf segment count mismatch: {len(leaf_seg_indices)} tree vs "
            f"{len(An_per_leaf_seg)} An array"
        )

        segments = self.plant.getSegments()
        for i, seg_idx in enumerate(leaf_seg_indices):
            child_node = segments[seg_idx].y
            # mol CO2/d -> mmol Suc/d
            An_nodes[child_node] = An_per_leaf_seg[i] * self.CO2_TO_SUC

        return An_nodes

    def _compute_fluxes(self, C_ST, An_source, temp_factor):
        """Compute loading, Rm, Rg, exudation, storage at given C_ST.

        Returns:
            loading, Rm_node, Rg_node, exud_node, storage_node — all np.array(N).
        """
        t = self.tree
        p = self.params
        N = t.n_nodes

        loading = np.zeros(N)
        Rm_node = np.zeros(N)
        Rg_node = np.zeros(N)
        exud_node = np.zeros(N)
        storage_node = np.zeros(N)

        for node in range(N):
            C = max(0.0, C_ST[node])

            # --- Loading (leaf source nodes) ---
            # At quasi-steady state, mesophyll C_meso adjusts so that
            # phloem loading Q_Fl ≈ Ag. The product inhibition exp(-C_ST*beta)
            # modulates the fraction exported (vs starch storage).
            # Loading = An * exp(-C_ST * beta), not capped by Vmax
            # (Vmax limits the rate in the ODE system, but C_meso compensates
            # at steady state).
            if self.loading_len[node] > 0 and An_source[node] > 0:
                inhibition = np.exp(-C * p.beta_loading)
                loading[node] = An_source[node] * inhibition

            # --- Usage: maintenance respiration ---
            C_eff = max(0.0, C - p.CSTimin)
            Q_Rmmax = (self.Q_Rmmax_base[node] + self.krm2_node[node] * C_eff) * temp_factor
            Fu_avail = C_eff / (C_eff + p.KMfu) if (C_eff + p.KMfu) > 0 else 0.0
            Fu = (Q_Rmmax + self.Q_Grmax_node[node]) * Fu_avail
            Rm_node[node] = min(Fu, Q_Rmmax)

            # --- Growth ---
            Rg_node[node] = max(0.0, min(Fu - Rm_node[node], self.Q_Grmax_node[node]))

            # --- Exudation (below-ground root nodes) ---
            if self.Q_Exudmax_coeff[node] > 0:
                C_delta = max(0.0, C_eff - p.C_soil)
                exud_node[node] = C_delta * self.Q_Exudmax_coeff[node]

            # --- Stem storage (starch accumulation in parenchyma) ---
            # Net storage = synthesis - hydrolysis:
            #   Q_store = k_S_ST * C_eff * V_par  (sucrose → starch)
            #   Q_hydro = kHyd_S_ST * C_starch     (starch → sucrose)
            # At quasi-steady state, C_starch = (k_S_ST / kHyd_S_ST) * C_eff
            # Net flux = k_S_ST * C_eff * V_par - kHyd_S_ST * C_starch * V_par
            # Substituting: Net = k_S_ST * C_eff * V_par * (1 - k_S_ST/kHyd_S_ST)
            # BUT: that gives 0 when k_S_ST == kHyd_S_ST (equilibrium).
            #
            # The key insight: starch ACCUMULATES over days/weeks because the
            # stem is continuously receiving new carbon from the phloem. The
            # storage rate represents the fraction of phloem-unloaded carbon
            # that goes to starch rather than back to sucrose. At vegetative
            # stage, synthesis > hydrolysis because the plant is building
            # reserves. We model net storage as:
            #   Q_store = k_S_ST * C_eff * V_parenchyma
            # with kHyd releasing starch only during grain fill (future).
            if self.storage_vol[node] > 0 and C_eff > 0:
                storage_node[node] = p.k_S_ST * C_eff * self.storage_vol[node]

        return loading, Rm_node, Rg_node, exud_node, storage_node

    def solve(self, An_per_leaf_seg, Tair_C=25.0,
              max_iter=500, tol=1e-3, alpha=0.6,
              balance_tol=0.02, sim_day=None, warm_start=None):
        """Solve for steady-state sucrose concentrations via Picard iteration
        with bisection-guided root collar concentration.

        Algorithm:
        Each iteration does:
        1. Forward sweep (tips→root): compute loading, Rm, Rg, exudation
           from current C_ST. Net surplus at each node flows to parent.
        2. Bisection update on C_base: surplus > 0 → raise C_lo, else
           raise C_hi. C_base = midpoint. Guaranteed to converge because
           surplus is monotonically decreasing in C_base (higher C →
           less loading via product inhibition, more sinks via Michaelis-Menten).
        3. Backward sweep (root→tips): anchor C_ST[root] = C_base.
           C_child = C_parent + flow / K_sugar. Damped update.
        4. Repeat until convergence (balance_error < balance_tol).

        Args:
            An_per_leaf_seg: np.array, mol CO2/d per leaf segment.
            Tair_C: Air temperature [C].
            max_iter: Maximum iterations.
            tol: Convergence tolerance on max|delta_C|.
            alpha: Damping factor (0 < alpha <= 1).
            balance_tol: Convergence tolerance on carbon balance error.
            sim_day: Simulation day (for seed reserve calculation). If None,
                     no seed reserve is applied.
            warm_start: Optional dict with 'C_ST_mean', 'C_ST_min', 'C_ST_max'
                from a previous day's solution. Used to initialize bisection
                bounds and C_base for faster convergence.

        Returns:
            dict with carbon partitioning results.
        """
        t = self.tree
        p = self.params
        N = t.n_nodes

        # Map An to nodes
        An_source = self._map_An_to_nodes(An_per_leaf_seg)
        total_An_mmol_suc = float(np.sum(An_source))

        # Seed reserve: maize kernel endosperm (~200 mg starch = ~585 umol
        # sucrose) mobilized over ~20 days. Provides 30-80 mmol Suc/d in
        # weeks 1-2, declining exponentially. Injected at root collar (node 0).
        # After day 25, reserves are depleted and contribution is negligible.
        seed_reserve_mmol = 0.0
        if sim_day is not None and sim_day <= 25:
            # Exponential depletion: half-life ~7 days
            # Peak rate at day 0: ~60 mmol Suc/d
            SEED_PEAK_RATE = 60.0   # mmol Suc/d at germination
            SEED_HALF_LIFE = 7.0    # days
            seed_reserve_mmol = SEED_PEAK_RATE * 2.0 ** (-sim_day / SEED_HALF_LIFE)
            An_source[0] += seed_reserve_mmol

        # Temperature factor for maintenance respiration
        temp_factor = p.Q10 ** ((Tair_C - p.TrefQ10) / 10.0)

        # Adaptive damping: when growth demand >> supply (young plants),
        # the balance is hypersensitive to C_base shifts. Lower alpha
        # stabilizes the Picard iteration.
        total_Grmax = float(np.sum(self.Q_Grmax_node))
        total_supply = total_An_mmol_suc + seed_reserve_mmol
        if total_supply > 0 and total_Grmax / total_supply > 2.0:
            alpha = min(alpha, 0.4)

        # Bisection bounds for root collar concentration
        if warm_start is not None and warm_start.get('C_ST_mean') is not None:
            ws_mean = warm_start['C_ST_mean']
            ws_min = warm_start['C_ST_min']
            ws_max = warm_start['C_ST_max']
            C_base = ws_mean
            C_lo = max(p.CSTimin + 0.001, ws_min * 0.8)
            C_hi = max(ws_max * 1.5, 3.0 * p.C_targ)
            max_iter = max(max_iter, 800)
        else:
            C_lo = p.CSTimin + 0.001
            C_hi = 3.0 * p.C_targ
            C_base = p.C_targ

        # Initialize C_ST
        C_ST = np.full(N, C_base)

        converged = False
        n_iter = 0
        max_delta = np.inf
        balance_error = np.inf
        best_balance = np.inf
        best_C_ST = C_ST.copy()
        best_loading = None
        best_Rm = None
        best_Rg = None
        best_exud = None
        best_storage = None

        for iteration in range(max_iter):
            n_iter = iteration + 1
            C_ST_old = C_ST.copy()

            # --- Forward sweep: compute fluxes and net flow to parent ---
            loading, Rm_node, Rg_node, exud_node, storage_node = self._compute_fluxes(
                C_ST, An_source, temp_factor
            )

            flow_to_parent = np.zeros(N)
            for node in t.reverse_topo_order:
                children_import = sum(
                    flow_to_parent[ch] for ch in t.children[node]
                )
                net = (loading[node] + children_import
                       - Rm_node[node] - Rg_node[node] - exud_node[node]
                       - storage_node[node])
                flow_to_parent[node] = net

            # --- Check global balance and update bisection ---
            surplus = flow_to_parent[0]
            total_loading = float(np.sum(loading))
            total_sinks = float(np.sum(Rm_node) + np.sum(Rg_node)
                                + np.sum(exud_node) + np.sum(storage_node))
            balance_denom = max(total_loading, total_sinks, 1e-6)
            balance_error = abs(surplus) / balance_denom

            # Track best solution
            if balance_error < best_balance:
                best_balance = balance_error
                best_C_ST = C_ST.copy()
                best_loading = loading.copy()
                best_Rm = Rm_node.copy()
                best_Rg = Rg_node.copy()
                best_exud = exud_node.copy()
                best_storage = storage_node.copy()

            if balance_error < balance_tol and max_delta < tol:
                converged = True
                break

            # Bisection update of root collar concentration
            if surplus > 0:
                # Loading > sinks → C too low → raise lower bound
                C_lo = max(C_lo, C_base)
            else:
                # Sinks > loading → C too high → lower upper bound
                C_hi = min(C_hi, C_base)

            C_base = 0.5 * (C_lo + C_hi)

            # --- Backward sweep with updated C_base ---
            C_ST_new = np.full(N, C_base)

            for node in t.topo_order:
                if node == 0:
                    continue
                parent = t.parent_of[node]
                K = self.K_sugar[node]
                if K > 0:
                    delta_C = flow_to_parent[node] / K
                    delta_C = np.clip(
                        delta_C, -0.5 * C_base, 0.5 * C_base
                    )
                    C_ST_new[node] = C_ST_new[parent] + delta_C
                else:
                    C_ST_new[node] = C_ST_new[parent]

            C_ST_new = np.clip(C_ST_new, 0.01, 5.0 * p.C_targ)

            # Damped update
            C_ST = (1 - alpha) * C_ST_old + alpha * C_ST_new

            max_delta = float(np.max(np.abs(C_ST - C_ST_old)))

            # Check if bisection has narrowed enough
            if (C_hi - C_lo) < 1e-5 and balance_error < balance_tol:
                converged = True
                break

        # Use best solution found
        if best_loading is not None:
            loading = best_loading
            Rm_node = best_Rm
            Rg_node = best_Rg
            exud_node = best_exud
            storage_node = best_storage
            C_ST = best_C_ST
        else:
            # Final flux evaluation
            loading, Rm_node, Rg_node, exud_node, storage_node = self._compute_fluxes(
                C_ST, An_source, temp_factor
            )

        return self._build_output(
            C_ST=C_ST,
            loading=loading,
            Rm_node=Rm_node,
            Rg_node=Rg_node,
            exud_node=exud_node,
            storage_node=storage_node,
            total_An_mmol_suc=total_An_mmol_suc,
            n_iter=n_iter,
            converged=converged,
            max_delta=max_delta,
            seed_reserve_mmol=seed_reserve_mmol,
        )

    def _build_output(self, C_ST, loading, Rm_node, Rg_node, exud_node,
                      storage_node, total_An_mmol_suc, n_iter, converged,
                      max_delta, seed_reserve_mmol=0.0):
        """Format solver results into the standard output dict."""
        t = self.tree

        # Aggregate by organ type
        Rm_leaf = Rm_stem = Rm_root = Rm_storage = 0.0
        Rg_leaf = Rg_stem = Rg_root = Rg_storage = 0.0
        total_stem_storage = 0.0

        segments = self.plant.getSegments()
        for seg_idx in range(t.n_segments):
            child_node = segments[seg_idx].y
            ot = int(t.organ_type[seg_idx])

            if ot == 4:    # leaf
                Rm_leaf += Rm_node[child_node]
                Rg_leaf += Rg_node[child_node]
            elif ot == 3:  # stem
                Rm_stem += Rm_node[child_node]
                Rg_stem += Rg_node[child_node]
                total_stem_storage += storage_node[child_node]
            elif ot == 2:  # root
                Rm_root += Rm_node[child_node]
                Rg_root += Rg_node[child_node]

        Rm_total = Rm_leaf + Rm_stem + Rm_root + Rm_storage
        Rg_total = Rg_leaf + Rg_stem + Rg_root + Rg_storage
        total_loading = float(np.sum(loading))
        total_exud = float(np.sum(exud_node))
        total_storage = float(np.sum(storage_node))
        growth = Rg_total

        # Carbon balance: at steady state, loading = sinks (transport conserves mass).
        # Sinks now include stem storage (starch accumulation).
        # The An surplus (An - loading) goes to leaf starch/storage.
        total_sinks = Rm_total + Rg_total + total_exud + total_storage
        # Balance error: how well does loading match sinks (should be near 0)
        balance_denom = max(total_loading, total_sinks, 1e-6)
        balance_error = abs(total_loading - total_sinks) / balance_denom
        starch_surplus = total_An_mmol_suc - total_loading  # leaf starch (not loaded)

        # Partitioning fractions (of total usage including storage)
        # Computed BEFORE unit conversion (fractions are unitless).
        total_usage = Rm_total + Rg_total + total_exud + total_storage
        if total_usage > 0:
            FR_leaf = (Rm_leaf + Rg_leaf) / total_usage
            FR_stem = (Rm_stem + Rg_stem + total_stem_storage) / total_usage
            FR_root = (Rm_root + Rg_root + total_exud) / total_usage
            FR_storage = 0.0  # storage is included in FR_stem
        else:
            FR_leaf = FR_stem = FR_root = FR_storage = 0.0

        # Convert respiration/growth fluxes from mmol Suc/d to mmol CO2/d.
        # 1 mmol Suc fully oxidised → 12 mmol CO2.
        # This matches the DVS solver's output units (all mmol CO2/d).
        # Exudation/dead root stay in mmol Suc/d (downstream expects sucrose
        # for kg C conversion via sucrose molar mass).
        S = self.SUC_TO_CO2

        return {
            'Rm_total_mmol': Rm_total * S,
            'Rm_leaf': Rm_leaf * S,
            'Rm_stem': Rm_stem * S,
            'Rm_root': Rm_root * S,
            'Rm_storage': Rm_storage * S,
            'Rg_total_mmol': Rg_total * S,
            'stem_storage_mmol': total_stem_storage * S,
            'FR_leaf': FR_leaf,
            'FR_stem': FR_stem,
            'FR_root': FR_root,
            'FR_storage': FR_storage,
            'root_resp_profile_mmol_d': np.array([Rm_root * S]),
            'root_exud_mmol_d': np.array([total_exud]),  # stays mmol Suc/d
            'root_dead_mmol_d': np.array([0.0]),          # stays mmol Suc/d
            'growth_mmol_d': growth * S,
            'carbon_balance_error': balance_error,
            'C_ST_mean': float(np.mean(C_ST)),
            'C_ST_min': float(np.min(C_ST)),
            'C_ST_max': float(np.max(C_ST)),
            'n_iterations': n_iter,
            'converged': converged,
            'max_delta': max_delta,
            'total_loading_mmol': total_loading * S,
            'starch_surplus_mmol': starch_surplus * S,
            'total_An_mmol_suc': total_An_mmol_suc,
            'seed_reserve_mmol': seed_reserve_mmol * S,
            'partitioning_source': 'quasi_steady_phloem',
        }


def solve_carbon_partitioning(plant, An_per_leaf_seg, Tair_C=25.0,
                              method='auto', day=55, warm_start=None):
    """High-level API for carbon partitioning.

    Args:
        plant: pb.MappedPlant (grown, with soil grid).
        An_per_leaf_seg: np.array, mol CO2/d per leaf segment.
        Tair_C: Air temperature [C].
        method: 'phloem', 'dvs', or 'auto' (tries phloem, falls back to DVS
                if carbon balance error > 10%).
        day: Simulation day (for DVS calculation).

    Returns:
        dict with carbon partitioning results.
    """
    from .dvs_partitioning import partition_carbon_dvs

    GPP_mmol = float(np.sum(An_per_leaf_seg)) * 1000.0  # mol -> mmol CO2/d

    if method == 'dvs':
        return partition_carbon_dvs(GPP_mmol, day, Tair_C=Tair_C)

    # Try quasi-steady phloem
    try:
        solver = QuasiSteadyPhloem(plant, sim_day=day)
        result = solver.solve(An_per_leaf_seg, Tair_C=Tair_C, sim_day=day,
                              warm_start=warm_start)

        if method == 'auto' and result['carbon_balance_error'] > 0.10:
            print(f"  Phloem balance error {result['carbon_balance_error']:.1%} > 10%, "
                  f"falling back to DVS")
            return partition_carbon_dvs(GPP_mmol, day, Tair_C=Tair_C)

        return result

    except Exception as e:
        if method == 'phloem':
            raise
        print(f"  Phloem solver failed ({e}), falling back to DVS")
        return partition_carbon_dvs(GPP_mmol, day, Tair_C=Tair_C)
