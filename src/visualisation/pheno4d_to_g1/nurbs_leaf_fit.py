"""Robust NURBS leaf-surface fit, vendored from Annika's main_NURBS.py.

The B-spline math here is a faithful copy of the MATLAB→Python port of
``fitLeafNurbsSurface_localWidth`` (Tobies / D.M. Spink NURBS Toolbox
workflow): open-uniform clamped knot vectors, Cox-de-Boor basis, a control-net
Laplacian regulariser, and a tensor-product least-squares solve. That math was
verified bit-for-bit against the MATLAB original (partition-of-unity to 1e-16,
exact clamped endpoints, column-major reshape).

Two label-free robustness fixes are layered on top so the fitter survives the
sparser, noisier candidates that the FP4D field segmentation produces (the
MATLAB version assumes ≥200-pt clean lab clouds):

  1. **Statistical outlier removal** before the fit — the MATLAB pipeline runs
     ``pcdenoise`` (Threshold 3, 30 neighbours); the original Python port
     dropped it. Re-added here as a kNN-distance filter.
  2. **Adaptive width binning** — fixed ``nWidthBins=50`` with a 20-pt-per-bin
     floor throws "too few valid u-bins" on ~500-pt leaves. ``nBins`` and the
     per-bin floor now scale with the point count.

Units are whatever ``P`` is in; the caller decides. ``fit_rms`` is the
point-to-surface RMS in the same units.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from sklearn.neighbors import NearestNeighbors


DEFAULTS = dict(
    p=3, q=3, nCtrlU=18, nCtrlV=7, lam=1e-2,
    left_percentile=2.0, right_percentile=98.0,
    min_points_per_leaf=120,
)


# ── faithful B-spline core (verified against bsplineBasisMatrix.m) ─────────


def open_uniform_knot_vector(nCtrl, p):
    nInternal = nCtrl - p - 1
    internal = (np.linspace(0.0, 1.0, nInternal + 2)[1:-1]
                if nInternal > 0 else np.array([]))
    return np.concatenate([np.zeros(p + 1), internal, np.ones(p + 1)])


def bspline_basis_matrix(u, knots, p):
    """Cox-de-Boor basis, m x nCtrl. Faithful port of bsplineBasisMatrix.m."""
    u = np.asarray(u, float).ravel()
    m = u.size
    nCtrl = len(knots) - p - 1
    eps = np.finfo(float).eps

    N0 = np.zeros((m, nCtrl + p))
    for i in range(nCtrl + p):
        N0[:, i] = ((u >= knots[i]) & (u < knots[i + 1])).astype(float)
    idxEnd = np.abs(u - 1.0) < 1e-12
    N0[idxEnd, :] = 0.0
    N0[idxEnd, nCtrl - 1] = 1.0

    Nprev = N0
    for d in range(1, p + 1):
        Ncurr = np.zeros((m, nCtrl + p - d))
        for i in range(nCtrl + p - d):
            denom1 = knots[i + d] - knots[i]
            denom2 = knots[i + d + 1] - knots[i + 1]
            term1 = np.zeros(m)
            term2 = np.zeros(m)
            if denom1 > eps:
                term1 = ((u - knots[i]) / denom1) * Nprev[:, i]
            if denom2 > eps:
                term2 = ((knots[i + d + 1] - u) / denom2) * Nprev[:, i + 1]
            Ncurr[:, i] = term1 + term2
        Nprev = Ncurr
    return Nprev[:, :nCtrl]


def _sub2ind_F(nU, nV, i, j):
    return j * nU + i  # column-major, matches MATLAB sub2ind + reshape


def build_control_net_laplacian(nCtrlU, nCtrlV):
    nCtrl = nCtrlU * nCtrlV
    rows, cols, vals = [], [], []
    r = -1
    for i in range(nCtrlU):
        for j in range(nCtrlV):
            cid = _sub2ind_F(nCtrlU, nCtrlV, i, j)
            neigh = []
            if i > 0:          neigh.append(_sub2ind_F(nCtrlU, nCtrlV, i - 1, j))
            if i < nCtrlU - 1: neigh.append(_sub2ind_F(nCtrlU, nCtrlV, i + 1, j))
            if j > 0:          neigh.append(_sub2ind_F(nCtrlU, nCtrlV, i, j - 1))
            if j < nCtrlV - 1: neigh.append(_sub2ind_F(nCtrlU, nCtrlV, i, j + 1))
            if len(neigh) >= 2:
                r += 1
                rows.append(r); cols.append(cid); vals.append(1.0)
                for nb in neigh:
                    rows.append(r); cols.append(nb); vals.append(-1.0 / len(neigh))
    return sp.csr_matrix((vals, (rows, cols)), shape=(r + 1, nCtrl))


# ── robustness layer ──────────────────────────────────────────────────────


def denoise_statistical(P, n_neighbors=30, std_ratio=3.0):
    """Statistical outlier removal (the ``pcdenoise`` step from main_NURBS.m).

    Drops points whose mean distance to their ``n_neighbors`` nearest
    neighbours exceeds ``mean + std_ratio * std`` over the cloud.
    """
    P = np.asarray(P, float)
    n = len(P)
    if n <= n_neighbors + 1:
        return P, np.ones(n, bool)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(P)
    dist, _ = nn.kneighbors(P)
    mean_d = dist[:, 1:].mean(axis=1)  # skip self
    thr = mean_d.mean() + std_ratio * mean_d.std()
    keep = mean_d <= thr
    return P[keep], keep


def _adaptive_bins(n_points):
    """Width-bin count and per-bin floor that scale with point count."""
    n_bins = int(np.clip(n_points // 30, 8, 50))
    min_bin = max(5, int(np.clip(n_points // (n_bins * 3), 5, 20)))
    return n_bins, min_bin


# ── main fit ──────────────────────────────────────────────────────────────


def fit_leaf_nurbs_surface(P, nCtrlU=18, nCtrlV=7, p=3, q=3, lam=1e-2,
                           nBins=None, leftPrct=2.0, rightPrct=98.0,
                           denoise=True):
    """Fit a B-spline (NURBS, w=1) surface to one leaf cloud.

    Mirrors ``fitLeafNurbsSurface_localWidth`` with the robustness layer above.

    Returns a dict: ``coefs`` (4, nCtrlU, nCtrlV), ``knots``, ``order``,
    ``fit_rms`` (point-to-surface RMS in P's units), ``P0``, ``n_used``.
    """
    P = np.asarray(P, float)
    P = P[np.isfinite(P).all(axis=1)]
    if denoise:
        P, _ = denoise_statistical(P)
    if P.shape[0] < 20:
        raise ValueError("Too few points for NURBS fit.")
    if nCtrlU <= p or nCtrlV <= q:
        raise ValueError("nCtrl must exceed degree.")

    if nBins is None:
        nBins, min_bin = _adaptive_bins(P.shape[0])
    else:
        min_bin = max(5, min(20, P.shape[0] // (nBins * 3) or 5))

    # 1) PCA orientation
    P0 = P.mean(axis=0)
    Pc = P - P0
    w, V = np.linalg.eigh(np.cov(Pc, rowvar=False))
    V = V[:, np.argsort(w)[::-1]]
    score = Pc @ V
    a = score[:, 0]   # along length -> u
    b = score[:, 1]   # across width -> v

    # 2) u along length
    aMin, aMax = a.min(), a.max()
    if abs(aMax - aMin) < np.finfo(float).eps:
        raise ValueError("Degenerate cloud: no length extent.")
    u = np.clip((a - aMin) / (aMax - aMin), 0.0, 1.0)

    # 3) local width(u) via percentile bins (adaptive floor)
    edges = np.linspace(0, 1, nBins + 1)
    uBin = 0.5 * (edges[:-1] + edges[1:])
    bLeft = np.full(nBins, np.nan)
    bRight = np.full(nBins, np.nan)
    for i in range(nBins):
        if i < nBins - 1:
            idx = (u >= edges[i]) & (u < edges[i + 1])
        else:
            idx = (u >= edges[i]) & (u <= edges[i + 1])
        if np.count_nonzero(idx) >= min_bin:
            vals = b[idx]
            bLeft[i] = np.percentile(vals, leftPrct)
            bRight[i] = np.percentile(vals, rightPrct)
    valid = np.isfinite(bLeft) & np.isfinite(bRight) & (bRight > bLeft)
    if np.count_nonzero(valid) < 4:
        raise ValueError(
            f"Too few valid u-bins for width parametrisation "
            f"({np.count_nonzero(valid)} valid of {nBins} bins, "
            f"floor {min_bin} pts/bin, {P.shape[0]} pts).")
    bLeftI = np.interp(u, uBin[valid], bLeft[valid])
    bRightI = np.interp(u, uBin[valid], bRight[valid])
    width = bRightI - bLeftI
    pos = width[width > 0]
    if pos.size == 0:
        raise ValueError("No positive local leaf width.")
    width[width < 0.02 * np.median(pos)] = 0.02 * np.median(pos)

    # 4) v local-normalised
    v = np.clip((b - bLeftI) / width, 0.0, 1.0)

    # 5-8) knots, basis, design matrix, regularised least squares
    ku = open_uniform_knot_vector(nCtrlU, p)
    kv = open_uniform_knot_vector(nCtrlV, q)
    Nu = bspline_basis_matrix(u, ku, p)
    Nv = bspline_basis_matrix(v, kv, q)
    n = P.shape[0]
    A = (Nu[:, :, None] * Nv[:, None, :]).reshape(n, nCtrlU * nCtrlV, order='F')
    L = build_control_net_laplacian(nCtrlU, nCtrlV)
    Asp = sp.csr_matrix(A)
    Nmat = (Asp.T @ Asp + lam * (L.T @ L)).tocsc()
    C = np.column_stack([spsolve(Nmat, A.T @ P[:, k]) for k in range(3)])

    Xc = C[:, 0].reshape(nCtrlU, nCtrlV, order='F')
    Yc = C[:, 1].reshape(nCtrlU, nCtrlV, order='F')
    Zc = C[:, 2].reshape(nCtrlU, nCtrlV, order='F')
    coefs = np.zeros((4, nCtrlU, nCtrlV))
    coefs[0] = Xc; coefs[1] = Yc; coefs[2] = Zc; coefs[3] = 1.0

    pred = A @ C
    resid = float(np.sqrt(np.mean(np.sum((pred - P) ** 2, axis=1))))
    return dict(coefs=coefs, knots=(ku, kv), order=(p + 1, q + 1),
                fit_rms=resid, P0=P0, n_used=int(n))


def nrbeval_grid(srf, nu=80, nv=24):
    """Evaluate the fitted surface on a (nu × nv × 3) grid."""
    ku, kv = srf['knots']
    p, q = srf['order'][0] - 1, srf['order'][1] - 1
    Nu = bspline_basis_matrix(np.linspace(0, 1, nu), ku, p)
    Nv = bspline_basis_matrix(np.linspace(0, 1, nv), kv, q)
    pts = np.empty((nu, nv, 3))
    for c in range(3):
        pts[:, :, c] = Nu @ srf['coefs'][c] @ Nv.T
    return pts


def to_nurbs_patch(srf):
    """Convert to a ``pb.NurbsPatch`` (CPlantBox round-trip)."""
    import plantbox as pb
    coefs = srf['coefs']
    nU, nV = coefs.shape[1], coefs.shape[2]
    cps = [[pb.Vector3d(float(coefs[0, i, j]), float(coefs[1, i, j]),
                        float(coefs[2, i, j])) for j in range(nV)]
           for i in range(nU)]
    ku, kv = srf['knots']
    return pb.NurbsPatch(cps, srf['order'][0] - 1, srf['order'][1] - 1,
                         list(map(float, ku)), list(map(float, kv)))
