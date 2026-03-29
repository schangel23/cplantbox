"""Post-process CPlantBox skeletons to fix tropism model limitations.

CPlantBox's tropism applies a continuous gravitropic curve from the base,
but real maize leaves have a stiff straight base (sheath region) with a
sharp bend at the collar/ligule, then a gentler droop after that.

This module straightens the base segments up to a configurable collar
point, producing more realistic leaf shapes without modifying CPlantBox.
"""

import numpy as np


def straighten_base(
    skeleton: np.ndarray,
    collar_frac: float = 0.3,
    blend_frac: float = 0.1,
) -> np.ndarray:
    """Straighten the base of a leaf skeleton up to the collar point.

    The base segments (0 to collar_frac along arc length) are projected
    onto a straight line from node[0] in the initial tangent direction.
    A smooth blend zone (collar_frac to collar_frac + blend_frac)
    transitions back to the original CPlantBox curve.

    Args:
        skeleton: (N, 3) leaf skeleton positions.
        collar_frac: fraction of arc length that stays straight (0-1).
        blend_frac: fraction of arc length for smooth transition (0-1).

    Returns:
        (N, 3) modified skeleton.
    """
    n = len(skeleton)
    if n < 4:
        return skeleton.copy()

    result = skeleton.copy()

    # Compute arc lengths
    diffs = np.diff(skeleton, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    total_len = np.sum(seg_lens)
    if total_len < 1e-6:
        return result

    cum_arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
    fracs = cum_arc / total_len

    # Initial tangent direction (average of first few segments for stability)
    n_avg = min(5, n - 1)
    tangent = skeleton[n_avg] - skeleton[0]
    tangent_len = np.linalg.norm(tangent)
    if tangent_len < 1e-6:
        return result
    tangent = tangent / tangent_len

    base = skeleton[0].copy()
    collar_end = collar_frac + blend_frac

    for i in range(1, n):
        t = fracs[i]
        if t <= collar_frac:
            # Fully straight: project onto tangent line
            result[i] = base + tangent * cum_arc[i]
        elif t < collar_end:
            # Blend zone: smooth transition from straight to original
            # Hermite smoothstep for C1 continuity
            blend_t = (t - collar_frac) / blend_frac
            blend_t = blend_t * blend_t * (3.0 - 2.0 * blend_t)  # smoothstep

            straight_pos = base + tangent * cum_arc[i]
            result[i] = straight_pos * (1.0 - blend_t) + skeleton[i] * blend_t
        # else: keep original CPlantBox position

    return result


def postprocess_organs(
    leaf_organs: list[dict],
    collar_frac: float = 0.3,
    blend_frac: float = 0.1,
) -> list[dict]:
    """Apply skeleton post-processing to all leaf organ dicts.

    Non-destructive: creates new organ dicts with modified skeletons.

    Args:
        leaf_organs: list of organ dicts from extract_organs_for_lofter()
        collar_frac: fraction of leaf that stays straight
        blend_frac: fraction for smooth transition

    Returns:
        list of organ dicts with straightened skeletons
    """
    result = []
    for organ in leaf_organs:
        new_organ = dict(organ)
        new_organ['skeleton'] = straighten_base(
            organ['skeleton'], collar_frac, blend_frac,
        )
        result.append(new_organ)
    return result
