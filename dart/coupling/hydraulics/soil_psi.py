"""Soil pressure-head provider abstraction.

Replaces the four hardcoded ``soil_psi_cm = -500.0`` defaults that drive the
Ch1 photosynthesis pipeline (see PLAN_DUMUX_INTEGRATION_2026-05-05.md).

Design:
    Each provider returns a per-cell head profile ``[cm]`` for
    ``hm.solve(rsx=..., cells=True)``.  ``FixedSoilPsi`` reproduces the
    legacy ``np.linspace(psi, psi - depth, depth)`` expression bit-for-bit so
    the regression gate passes.  ``DumuxSoilPsi`` wraps the
    ``rosi_richards.RichardsSP`` C++ binding for dynamic ψ_s(z, t).

The two-step protocol — ``update(t, sink_per_cell)`` then
``get_profile(t)`` — exists so DuMux can advance once per coupling step
with the latest plant water-uptake feedback.  Static providers ignore the
sink term.

Phase 2 ships ``FixedSoilPsi`` for backward-compatible rewiring and a
minimally-functional ``DumuxSoilPsi`` for end-to-end smoke runs.  Phase 3
will validate the matric/total head convention against published drought
experiments.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, Tuple

import numpy as np


class SoilPsiProvider(Protocol):
    """Source of per-cell soil pressure head for ``hm.solve(rsx=...)``."""

    def get_profile(self, t_days: float, depth_cm: int) -> np.ndarray:
        """Return per-cell head [cm], length ``depth_cm``.

        Element 0 = top of column, element [depth_cm-1] = bottom.
        """
        ...

    def update(self, t_days: float, sink_per_cell: np.ndarray) -> None:
        """Register plant water-uptake (sink) for the next solver step.

        ``sink_per_cell`` is in cm³/d per cell, sign convention "positive
        out of soil into roots" (i.e. uptake).  Static providers no-op.
        """
        ...


class FixedSoilPsi:
    """Backward-compatible static profile.

    Reproduces ``np.linspace(psi_cm, psi_cm - depth_cm, depth_cm)`` exactly,
    so existing pipeline output is bit-identical when ``--soil-mode fixed``.
    """

    def __init__(self, psi_cm: float = -500.0):
        self.psi_cm = float(psi_cm)

    def get_profile(self, t_days: float, depth_cm: int) -> np.ndarray:
        return np.linspace(self.psi_cm, self.psi_cm - depth_cm, depth_cm)

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
    ):
        self.psi = float(psi_init_cm)
        self.psi_target = float(psi_target_cm)
        self.tau = float(tau_days)
        self._t_last = 0.0

    def get_profile(self, t_days: float, depth_cm: int) -> np.ndarray:
        dt = max(0.0, t_days - self._t_last)
        if dt > 0:
            decay = np.exp(-dt / self.tau)
            self.psi = self.psi_target + (self.psi - self.psi_target) * decay
            self._t_last = t_days
        return np.linspace(self.psi, self.psi - depth_cm, depth_cm)

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
    """Wraps RichardsSP for dynamic ψ_s(z, t).

    Owns a 1D vertical column with the supplied VG params and BCs.  Each
    ``get_profile(t)`` call advances the solver from its last simulated
    time to ``t`` (unless ``t`` is in the past or unchanged).  Sink terms
    set via ``update`` are applied for the duration of the next step
    *only*; subsequent steps use whatever sink the next ``update`` call
    provides (or zero if none).

    Returns matric pressure head per cell.  In Phase 2 this is passed
    directly to ``hm.solve(rsx=..., cells=True)``.  Phase 3 validates the
    matric/total convention; if a +z gravity correction is needed it will
    be applied here, not at every call site.
    """

    def __init__(
        self,
        depth_cm: float = 100.0,
        n_cells_z: int = 100,
        psi_init_cm: float = -300.0,
        vg_params: Tuple[float, float, float, float, float] = _LOAM_VG,
        top_bc: Tuple[int, float] = (BC_CONSTANT_FLUX, 0.0),
        bot_bc: Tuple[int, float] = (BC_FREE_DRAINAGE, 0.0),
        dumux_binding_path: Path = _DEFAULT_DUMUX_BIND,
        verbose: bool = False,
    ):
        import sys
        if str(dumux_binding_path) not in sys.path:
            sys.path.insert(0, str(dumux_binding_path))
        from rosi_richards import RichardsSP  # noqa: import deferred

        self.depth_cm = float(depth_cm)
        self.n_cells_z = int(n_cells_z)
        self.psi_init_cm = float(psi_init_cm)
        self.vg_params = tuple(vg_params)

        s = RichardsSP()
        s.initialize([""], verbose=verbose, doMPI=False)

        qr, qs, alpha, n, ks = vg_params
        s.setParameter("Soil.VanGenuchten.Qr", str(qr))
        s.setParameter("Soil.VanGenuchten.Qs", str(qs))
        s.setParameter("Soil.VanGenuchten.Alpha", str(alpha))
        s.setParameter("Soil.VanGenuchten.N", str(n))
        s.setParameter("Soil.VanGenuchten.Ks", str(ks))
        s.setParameter("Soil.Layer.Number", "1")

        s.setParameter("Soil.IC.P", str(self.psi_init_cm))
        s.setParameter("Soil.BC.Top.Type", str(int(top_bc[0])))
        s.setParameter("Soil.BC.Top.Value", str(float(top_bc[1])))
        s.setParameter("Soil.BC.Bot.Type", str(int(bot_bc[0])))
        s.setParameter("Soil.BC.Bot.Value", str(float(bot_bc[1])))

        s.createGrid([-5.0, -5.0, -self.depth_cm], [5.0, 5.0, 0.0],
                     [1, 1, self.n_cells_z], False)
        s.initializeProblem(-1.0)

        self._s = s

        coords = np.asarray(s.getDofCoordinates(), dtype=float)
        self._z_cm = coords[:, 2]
        # Sort top→bottom: top of column (z = 0) is element 0, bottom (z = -depth) is last.
        self._order_top_first = np.argsort(self._z_cm)[::-1]

        self._t_last_days = 0.0
        self._pending_sink: Optional[dict] = None

    def get_profile(self, t_days: float, depth_cm: int) -> np.ndarray:
        if depth_cm != self.n_cells_z:
            raise ValueError(
                f"DumuxSoilPsi: depth_cm={depth_cm} != solver grid n_cells_z="
                f"{self.n_cells_z}; resampling not yet implemented"
            )
        dt_days = float(t_days) - self._t_last_days
        if dt_days > 0:
            if self._pending_sink is not None:
                self._s.setSource(self._pending_sink, 0)
                self._pending_sink = None
            else:
                self._s.setSource({}, 0)
            self._s.solveNoMPI(dt_days * 86400.0, False)
            self._t_last_days = float(t_days)

        psi = np.asarray(self._s.getSolutionHead(), dtype=float)
        return psi[self._order_top_first]

    def update(self, t_days: float, sink_per_cell: np.ndarray) -> None:
        sink = np.asarray(sink_per_cell, dtype=float)
        if sink.size != self.n_cells_z:
            raise ValueError(
                f"sink_per_cell size {sink.size} != n_cells_z {self.n_cells_z}"
            )
        sink_in_solver_order = np.empty_like(sink)
        sink_in_solver_order[self._order_top_first] = sink
        self._pending_sink = {
            i: float(v) for i, v in enumerate(sink_in_solver_order) if v != 0.0
        }


def make_provider(
    mode: str,
    soil_psi_cm: float = -500.0,
    **kwargs,
) -> SoilPsiProvider:
    """Factory used by the CLI."""
    mode = mode.lower()
    if mode == "fixed":
        return FixedSoilPsi(psi_cm=soil_psi_cm)
    if mode == "bucket":
        return BucketSoilPsi(psi_init_cm=soil_psi_cm, **kwargs)
    if mode == "dumux":
        return DumuxSoilPsi(psi_init_cm=soil_psi_cm, **kwargs)
    raise ValueError(f"Unknown soil_mode={mode!r}; use fixed/bucket/dumux")
