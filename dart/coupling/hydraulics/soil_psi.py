"""Soil pressure-head provider abstraction.

Replaces the four hardcoded ``soil_psi_cm = -500.0`` defaults that drive the
Ch1 photosynthesis pipeline (see PLAN_DUMUX_INTEGRATION_2026-05-05.md).

Indexing convention (Phase 3.5+): each provider returns a flat array
indexed by **CPlantBox cellidx** — the linear index produced by
``MappedSegments::soil_index_(x, y, z)``:

    cellidx = floor(iz) * nx * ny + floor(iy) * nx + floor(ix)

with ``ix = (x - min_b.x) / (max_b.x - min_b.x) * cell_number.x`` etc. For
a 1×1×nz vertical column (the legacy default) this means **cellidx 0 is
the bottom cell** (z = -depth) and cellidx nz-1 is the top cell (z = 0).
This is the convention ``hm.solve(rsx=psi, cells=True)`` expects when
the plant's seg→cell mapping is wired via
``MappedSegments::setRectangularGrid`` (the upstream-canonical pattern).

Phase 0–3 used a hand-rolled ``_picker`` lambda + ``setSoilGrid`` that
mapped z=0 → cellidx 0 (top first). Phase 3.5 retired that workaround;
``FixedSoilPsi``/``BucketSoilPsi`` now return the linspace gradient
**reversed** so the physics remains bit-identical (top of column still
sees ``psi_cm``, bottom sees ``psi_cm - depth_cm``) under the new
indexing.

The two-step protocol — ``update(t, sink_per_cell)`` then
``get_profile(t)`` — exists so DuMux can advance once per coupling step
with the latest plant water-uptake feedback. Static providers ignore the
sink term.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, Sequence, Tuple

import numpy as np


def _cellidx_from_xyz(
    coords_cm: np.ndarray,
    min_b_cm: Sequence[float],
    max_b_cm: Sequence[float],
    cell_number: Sequence[int],
) -> np.ndarray:
    """Replicate ``MappedSegments::soil_index_`` for a batch of xyz centroids.

    Returns linear cellidx for each row of ``coords_cm`` (shape (N,3) in cm).
    """
    nx, ny, nz = cell_number
    minx, miny, minz = min_b_cm
    maxx, maxy, maxz = max_b_cm
    wx, wy, wz = (maxx - minx, maxy - miny, maxz - minz)
    ix = np.floor((coords_cm[:, 0] - minx) / wx * nx).astype(int)
    iy = np.floor((coords_cm[:, 1] - miny) / wy * ny).astype(int)
    iz = np.floor((coords_cm[:, 2] - minz) / wz * nz).astype(int)
    # DOF centroids should lie strictly interior to the grid; clamp guards
    # against floating-point edge cases at the (max_b) face.
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    iz = np.clip(iz, 0, nz - 1)
    return iz * nx * ny + iy * nx + ix


class SoilPsiProvider(Protocol):
    """Source of per-cell soil pressure head for ``hm.solve(rsx=...)``."""

    n_cells_total: int

    def get_profile(self, t_days: float, depth_cm: Optional[int] = None) -> np.ndarray:
        """Return per-cell head [cm], length ``n_cells_total``.

        Indexing: CPlantBox cellidx (see module docstring). ``depth_cm`` is
        retained as a length-validation kwarg; if supplied, must equal
        ``self.n_cells_total``.
        """
        ...

    def update(self, t_days: float, sink_per_cell: np.ndarray) -> None:
        """Register plant water-uptake (sink) for the next solver step.

        ``sink_per_cell`` is in cm³/d per cell, **negative = water leaving
        soil into roots** — same sign convention as
        ``RichardsSP.setSource()``: ``sum(sink_per_cell) ≈ -transpiration``.
        Indexed by CPlantBox cellidx, length ``n_cells_total``.
        Static providers no-op.
        """
        ...


def _validate_n_cells(n_cells_total: int, depth_cm: Optional[int]) -> int:
    if depth_cm is None:
        return n_cells_total
    n = int(depth_cm)
    if n != n_cells_total:
        raise ValueError(
            f"depth_cm={n} != provider n_cells_total={n_cells_total}; "
            "the legacy depth_cm kwarg is the array-length validator under "
            "the new cellidx convention, not a separate dimension"
        )
    return n


class FixedSoilPsi:
    """Backward-compatible static profile under the cellidx convention.

    Returns ``np.linspace(psi_cm - depth_cm, psi_cm, depth_cm)`` so cellidx
    0 (= bottom of a 1×1×depth_cm column) sees ``psi_cm - depth_cm`` and
    cellidx depth_cm-1 (= top) sees ``psi_cm``. **Physically identical** to
    the legacy top-first ``np.linspace(psi_cm, psi_cm - depth, depth)``
    expression once paired with ``setRectangularGrid`` (which inverts z
    relative to the old ``_picker``); the array is just reversed.
    """

    def __init__(self, psi_cm: float = -500.0, n_cells: int = 100):
        self.psi_cm = float(psi_cm)
        self.n_cells_total = int(n_cells)

    def get_profile(self, t_days: float, depth_cm: Optional[int] = None) -> np.ndarray:
        n = _validate_n_cells(self.n_cells_total, depth_cm)
        return np.linspace(self.psi_cm - n, self.psi_cm, n)

    def update(self, t_days: float, sink_per_cell: np.ndarray) -> None:
        return  # static


class BucketSoilPsi:
    """Lumped 1D bucket — exponential decay toward a target ψ.

    Simple fallback if DuMux is unavailable.  ``tau_days`` is the e-folding
    time of the soil drying response; ``psi_target_cm`` is the asymptote.
    Sink terms accelerate drying linearly (pulled out via a calibrated mass
    factor; not physical, just a smoke-test fallback).
    """

    def __init__(
        self,
        psi_init_cm: float = -300.0,
        psi_target_cm: float = -1500.0,
        tau_days: float = 30.0,
        n_cells: int = 100,
    ):
        self.psi = float(psi_init_cm)
        self.psi_target = float(psi_target_cm)
        self.tau = float(tau_days)
        self.n_cells_total = int(n_cells)
        self._t_last = 0.0

    def get_profile(self, t_days: float, depth_cm: Optional[int] = None) -> np.ndarray:
        n = _validate_n_cells(self.n_cells_total, depth_cm)
        dt = max(0.0, t_days - self._t_last)
        if dt > 0:
            decay = np.exp(-dt / self.tau)
            self.psi = self.psi_target + (self.psi - self.psi_target) * decay
            self._t_last = t_days
        return np.linspace(self.psi - n, self.psi, n)

    def update(self, t_days: float, sink_per_cell: np.ndarray) -> None:
        return  # bucket is too coarse to honour spatial sink


# Default DuMux build location — local Python 3.14 build.
_DEFAULT_DUMUX_BIND = Path(
    "/home/lukas/PHD/dumux-build/dumux/dumux-rosi/build-cmake/cpp/python_binding"
)

# Loam VG params: theta_r, theta_s, alpha [1/cm], n, Ksat [cm/d]
_LOAM_VG = (0.08, 0.43, 0.04, 1.6, 50.0)

# BC enums (cpp/soil_richards/richardsproblem.hh)
BC_CONSTANT_PRESSURE = 1
BC_CONSTANT_FLUX = 2
BC_ATMOSPHERIC = 4
BC_FREE_DRAINAGE = 5


class DumuxSoilPsi:
    """Wraps RichardsSP for dynamic ψ_s(x, y, z, t) on a 3D rectangular grid.

    The canonical-3D ctor takes ``(min_b, max_b, cell_number)`` triplets
    (cm, cm, ints); the legacy 1D-vertical ctor with ``(depth_cm,
    n_cells_z, col_half_width_cm)`` is still accepted and translates to
    ``min_b=(-w,-w,-depth)``, ``max_b=(w,w,0)``, ``cell_number=(1,1,nz)``.

    Returns matric pressure head per cell, indexed by CPlantBox cellidx
    (see module docstring). Internally builds a permutation from DuMux DOF
    order to cellidx order using the same formula CPlantBox uses for
    ``MappedSegments::setRectangularGrid``.

    Sink terms set via ``update`` are applied for the duration of the next
    step *only*; subsequent steps use whatever sink the next ``update``
    call provides (or zero if none).
    """

    def __init__(
        self,
        # Canonical 3D ctor — preferred:
        min_b: Optional[Sequence[float]] = None,
        max_b: Optional[Sequence[float]] = None,
        cell_number: Optional[Sequence[int]] = None,
        periodic: bool = False,
        # Common params:
        psi_init_cm: float = -300.0,
        vg_params: Tuple[float, float, float, float, float] = _LOAM_VG,
        top_bc: Tuple[int, float] = (BC_CONSTANT_FLUX, 0.0),
        bot_bc: Tuple[int, float] = (BC_FREE_DRAINAGE, 0.0),
        dumux_binding_path: Path = _DEFAULT_DUMUX_BIND,
        verbose: bool = False,
        # Legacy 1D-vertical kwargs (translated to (min_b, max_b, cell_number)
        # if any are supplied):
        depth_cm: Optional[float] = None,
        n_cells_z: Optional[int] = None,
        col_half_width_cm: Optional[float] = None,
    ):
        # Resolve grid spec: legacy kwargs take precedence if supplied,
        # else fall back to canonical (min_b, max_b, cell_number), else
        # default to a 1×1×100 column.
        legacy_supplied = any(v is not None for v in (depth_cm, n_cells_z, col_half_width_cm))
        canonical_supplied = any(v is not None for v in (min_b, max_b, cell_number))
        if legacy_supplied and canonical_supplied:
            raise ValueError(
                "DumuxSoilPsi: pass either (min_b, max_b, cell_number) OR "
                "the legacy (depth_cm, n_cells_z, col_half_width_cm) kwargs, "
                "not both"
            )
        if legacy_supplied:
            d = float(depth_cm if depth_cm is not None else 100.0)
            nz = int(n_cells_z if n_cells_z is not None else 100)
            w = float(col_half_width_cm if col_half_width_cm is not None else 5.0)
            min_b = (-w, -w, -d)
            max_b = (w, w, 0.0)
            cell_number = (1, 1, nz)
        else:
            if min_b is None:
                min_b = (-5.0, -5.0, -100.0)
            if max_b is None:
                max_b = (5.0, 5.0, 0.0)
            if cell_number is None:
                cell_number = (1, 1, 100)

        self.min_b = tuple(float(v) for v in min_b)
        self.max_b = tuple(float(v) for v in max_b)
        self.cell_number = tuple(int(v) for v in cell_number)
        self.n_cells_total = int(np.prod(self.cell_number))
        self.psi_init_cm = float(psi_init_cm)
        self.vg_params = tuple(vg_params)
        self.periodic = bool(periodic)

        # Backward-compat exposure (old tests/scripts read these):
        self.depth_cm = float(self.max_b[2] - self.min_b[2])
        self.n_cells_z = int(self.cell_number[2])
        self.col_half_width_cm = float(0.5 * (self.max_b[0] - self.min_b[0]))

        import sys
        if str(dumux_binding_path) not in sys.path:
            sys.path.insert(0, str(dumux_binding_path))
        from rosi_richards import RichardsSP  # noqa: import deferred

        s = RichardsSP()
        s.initialize([""], verbose=verbose, doMPI=False)

        qr, qs, alpha, n_vg, ks = self.vg_params
        s.setParameter("Soil.VanGenuchten.Qr", str(qr))
        s.setParameter("Soil.VanGenuchten.Qs", str(qs))
        s.setParameter("Soil.VanGenuchten.Alpha", str(alpha))
        s.setParameter("Soil.VanGenuchten.N", str(n_vg))
        s.setParameter("Soil.VanGenuchten.Ks", str(ks))
        s.setParameter("Soil.Layer.Number", "1")

        s.setParameter("Soil.IC.P", str(self.psi_init_cm))
        s.setParameter("Soil.BC.Top.Type", str(int(top_bc[0])))
        s.setParameter("Soil.BC.Top.Value", str(float(top_bc[1])))
        s.setParameter("Soil.BC.Bot.Type", str(int(bot_bc[0])))
        s.setParameter("Soil.BC.Bot.Value", str(float(bot_bc[1])))

        # DuMux uses SI units internally — coordinates in metres. The
        # richards.py Python wrapper converts cm → m via /100 (richards.py:80
        # and 92); we bypass the wrapper, so do the conversion here. Pressure
        # heads, however, remain in cm throughout (Soil.IC.P, Soil.BC.*.Value,
        # getSolutionHead all cm; richardsproblem.hh:395 converts internal Pa
        # → cm head via toHead_). Volumes therefore are also in m³, which is
        # what scv.volume() returns at richardsproblem.hh:391 to convert
        # source_ [kg/s per element] → [kg/(m³·s)].
        min_b_m = [v / 100.0 for v in self.min_b]
        max_b_m = [v / 100.0 for v in self.max_b]
        s.createGrid(min_b_m, max_b_m, list(self.cell_number), self.periodic)
        s.initializeProblem(-1.0)

        self._s = s

        # Build cellidx ↔ DOF permutation. DuMux's getDofCoordinates returns
        # element centroids in DOF order (in metres). For each DOF compute
        # the CPlantBox-formula cellidx; invert to map cellidx → DOF.
        # For sequential SPGrid runs gIdx == eIdx == DOF order, so setSource
        # keys (which expect gIdx) are simply DOF indices.
        coords_m = np.asarray(s.getDofCoordinates(), dtype=float)
        coords_cm = coords_m * 100.0
        self._cellidx_per_dof = _cellidx_from_xyz(
            coords_cm, self.min_b, self.max_b, self.cell_number,
        )
        if self._cellidx_per_dof.size != self.n_cells_total:
            raise RuntimeError(
                f"DumuxSoilPsi: getDofCoordinates returned "
                f"{self._cellidx_per_dof.size} DOFs but cell_number="
                f"{self.cell_number} → n_cells_total={self.n_cells_total}"
            )
        # Sanity: cellidx_per_dof must be a permutation of 0..N-1.
        if not np.array_equal(np.sort(self._cellidx_per_dof),
                              np.arange(self.n_cells_total)):
            raise RuntimeError(
                "DumuxSoilPsi: cellidx_per_dof is not a permutation of "
                "0..N-1; CPlantBox/DuMux indexing assumption violated"
            )
        self._dof_for_cellidx = np.empty(self.n_cells_total, dtype=int)
        self._dof_for_cellidx[self._cellidx_per_dof] = np.arange(self.n_cells_total)

        self._t_last_days = 0.0
        self._pending_sink: Optional[dict] = None

    def get_profile(self, t_days: float, depth_cm: Optional[int] = None) -> np.ndarray:
        _validate_n_cells(self.n_cells_total, depth_cm)
        dt_days = float(t_days) - self._t_last_days
        if dt_days > 0:
            if self._pending_sink is not None:
                self._s.setSource(self._pending_sink, 0)
                self._pending_sink = None
            else:
                self._s.setSource({}, 0)
            self._s.solveNoMPI(dt_days * 86400.0, False)
            self._t_last_days = float(t_days)

        psi_dof = np.asarray(self._s.getSolutionHead(), dtype=float)
        # Reorder DOF → cellidx. psi_cellidx[i] = psi at CPlantBox cellidx i.
        return psi_dof[self._dof_for_cellidx]

    def get_water_volume_cm3(self) -> float:
        """Total domain water volume [cm³].

        The C++ ``RichardsSP::getWaterVolume`` binding returns a volume
        in **m³** (DuMux internal units). The Python ``richards.py``
        wrapper applies ``* 1e6`` to give cm³; we bypass the wrapper, so
        this helper does the conversion. Used by the Gate Ch1.PMDM.3
        conservation test to compare ΔW_soil against
        ``integrated_rwu_cm3`` (which is cm³ throughout the rest of the
        coupling stack).
        """
        return float(self._s.getWaterVolume()) * 1.0e6

    def update(self, t_days: float, sink_per_cell: np.ndarray) -> None:
        sink = np.asarray(sink_per_cell, dtype=float)
        if sink.size != self.n_cells_total:
            raise ValueError(
                f"sink_per_cell size {sink.size} != n_cells_total "
                f"{self.n_cells_total}"
            )
        # Sink is delivered in cm³/d (CPlantBox convention). The C++
        # RichardsSP::setSource expects kg/s (richardsproblem.hh:479). The
        # Python wrapper richards.py:setSource performs this conversion
        # (value / 86400 / 1000), but we call the raw C++ binding —
        # replicate the conversion here. 1 cm³ water = 1 g = 1e-3 kg;
        # 1 day = 86400 s.
        cm3_per_day_to_kg_per_s = 1.0 / 86400.0 / 1000.0
        # sink is indexed by CPlantBox cellidx. setSource keys are gIdx
        # (DuMux global element index), which equals DOF index for
        # sequential SPGrid; map via _dof_for_cellidx.
        self._pending_sink = {
            int(self._dof_for_cellidx[i]): float(v) * cm3_per_day_to_kg_per_s
            for i, v in enumerate(sink) if v != 0.0
        }


def make_provider(
    mode: str,
    soil_psi_cm: float = -500.0,
    **kwargs,
) -> SoilPsiProvider:
    """Factory used by the CLI."""
    mode = mode.lower()
    if mode == "fixed":
        return FixedSoilPsi(psi_cm=soil_psi_cm, **kwargs)
    if mode == "bucket":
        return BucketSoilPsi(psi_init_cm=soil_psi_cm, **kwargs)
    if mode == "dumux":
        return DumuxSoilPsi(psi_init_cm=soil_psi_cm, **kwargs)
    raise ValueError(f"Unknown soil_mode={mode!r}; use fixed/bucket/dumux")


def push_rwu_sink_to_provider(
    hm,
    sim_time: float,
    p_s: np.ndarray,
    provider: SoilPsiProvider,
    n_cells: Optional[int] = None,
    verbose: bool = False,
    *,
    depth_cm: Optional[int] = None,  # legacy alias for n_cells (1D-vertical naming)
) -> dict:
    """Aggregate per-segment radial fluxes after ``hm.solve(...)`` and
    push them to the soil provider as a per-cell sink.

    Closes the soil↔plant water loop. Mirrors the canonical pattern in
    ``dumux-rosi/python/coupled/coupled_c11.py:81-103``: read xylem
    pressures, call ``hm.soil_fluxes`` for the seg→cell aggregation,
    feed the resulting ``{cell_idx: cm³/d}`` dict (negative = uptake)
    to the provider's ``update``.

    Static providers (``FixedSoilPsi``, ``BucketSoilPsi``) ignore sinks,
    so the aggregation is skipped for them — both for performance and
    to keep ``--soil-mode fixed`` runs bit-identical with pre-RWU code.

    Returns the raw fluxes dict for diagnostics (empty dict for static
    providers).
    """
    if isinstance(provider, (FixedSoilPsi, BucketSoilPsi)):
        return {}

    if n_cells is None:
        if depth_cm is not None:
            n_cells = int(depth_cm)
        else:
            n_cells = int(getattr(provider, "n_cells_total"))

    # PhotosynthesisPython overrides radial_fluxes() to a no-arg accessor for
    # the cached self.outputFlux (per-segment cm³/d, including leaf
    # transpiration). The parent HydraulicModel.soil_fluxes(t, rx, rsx)
    # breaks against that override. Detect the case and aggregate manually:
    # only root segments (organType == 2) with cellIdx >= 0 contribute to
    # the soil sink; the C++ sumSegFluxes guards against non-root segments
    # spuriously mapped into a soil cell (e.g. a basal stem at z ≈ 0), so
    # we mirror that filter in Python.
    try:
        rx = np.asarray(hm.get_water_potential(), dtype=float)
        fluxes = hm.soil_fluxes(float(sim_time), rx, p_s)  # {cell_idx: cm³/d}
    except TypeError:
        fluxes_per_seg = np.asarray(hm.radial_fluxes(), dtype=float)
        organ_types = np.asarray(hm.ms.organTypes, dtype=int)
        seg2cell = hm.ms.seg2cell
        fluxes = {}
        for seg_idx, cell_idx in seg2cell.items():
            if cell_idx < 0 or int(organ_types[seg_idx]) != 2:
                continue
            fluxes[cell_idx] = fluxes.get(cell_idx, 0.0) + float(fluxes_per_seg[seg_idx])

    if verbose:
        sum_flux = sum(fluxes.values())
        try:
            transp = float(np.sum(hm.get_transpiration()))  # cm³/d
            off_pct = 100.0 * abs(sum_flux + transp) / max(abs(transp), 1e-12)
            print(f"  RWU sink: ∑fluxes={sum_flux:.4g}, "
                  f"-transp={-transp:.4g} cm³/d, off={off_pct:.2f}%")
        except Exception:
            print(f"  RWU sink: ∑fluxes={sum_flux:.4g} cm³/d "
                  f"(transp readback failed)")

    sink = np.zeros(n_cells, dtype=float)
    for cell_idx, f in fluxes.items():
        if 0 <= cell_idx < n_cells:
            sink[cell_idx] += float(f)
    provider.update(float(sim_time), sink)
    return fluxes
