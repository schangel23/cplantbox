"""Generate CPlantBox skeleton training data for the neural surrogate.

Runs CPlantBox with sampled parameters, extracts per-leaf skeletons
(xyz + width), and stores the results in an HDF5 dataset.
"""

import copy
import logging
import math
import multiprocessing as mp
import os
import tempfile
import traceback
from pathlib import Path

import h5py
import numpy as np
import xml.etree.ElementTree as ET

from .param_sampler import (
    LEAF_PARAM_NAMES,
    N_LEAF_PARAMS,
    N_PARAMS,
    N_POSITIONS,
    STEM_PARAM_NAMES,
    _flatten_params,
    load_prior_bounds,
    sample_params,
)

logger = logging.getLogger(__name__)

N_MAX = 64  # maximum nodes per leaf skeleton (zero-padded)


def _apply_params_to_xml(template_path: str, params: dict, out_path: str) -> None:
    """Write a modified XML with the sampled parameter values.

    Modifies the calibrated XML in-place:
    - Per-leaf subType parameters (lmax, Width_blade, theta, tropismS,
      tropismAge, r, areaMax) for positions 0-10.
    - Stem internode length (ln).

    Deformation parameters (wave_normal_amp, twist_max, curl_amp,
    edge_ruffle_amp, fold_amp) are not stored in the XML; they are
    passed directly to the lofter at extraction time.
    """
    tree = ET.parse(template_path)
    root = tree.getroot()

    for pos in range(N_POSITIONS):
        sub_type = pos + 2  # leaf subtypes start at 2
        pos_params = params.get(pos, {})
        leaf_elem = root.find(f".//leaf[@subType='{sub_type}']")
        if leaf_elem is None:
            continue

        xml_params = {
            "lmax": pos_params.get("lmax"),
            "Width_blade": pos_params.get("Width_blade"),
            "theta": pos_params.get("theta"),
            "tropismS": pos_params.get("tropismS"),
            "tropismAge": pos_params.get("tropismAge"),
            "r": pos_params.get("r"),
            "areaMax": pos_params.get("areaMax"),
        }
        for pname, pval in xml_params.items():
            if pval is None:
                continue
            elem = leaf_elem.find(f".//parameter[@name='{pname}']")
            if elem is not None:
                elem.set("value", str(pval))
            else:
                new = ET.SubElement(leaf_elem, "parameter")
                new.set("name", pname)
                new.set("value", str(pval))

    # Stem internode length
    stem_ln = params.get("stem_ln")
    if stem_ln is not None:
        stem_elem = root.find(".//stem[@subType='1']")
        if stem_elem is not None:
            ln_elem = stem_elem.find(".//parameter[@name='ln']")
            if ln_elem is not None:
                ln_elem.set("value", str(stem_ln))

    tree.write(out_path, encoding="utf-8", xml_declaration=True)


def _extract_leaf_skeletons(plant, n_max: int = N_MAX) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract leaf skeletons from a grown plant.

    Returns:
        skeletons: (N_POSITIONS, n_max, 4) — xyz + half-width, zero-padded.
        masks: (N_POSITIONS, n_max) bool — True for valid nodes.
        n_leaves: number of leaves that emerged.
    """
    import plantbox as pb

    skeletons = np.zeros((N_POSITIONS, n_max, 4), dtype=np.float32)
    masks = np.zeros((N_POSITIONS, n_max), dtype=bool)

    leaf_organs = [o for o in plant.getOrgans() if o.organType() == pb.OrganTypes.leaf]

    n_leaves = 0
    for organ in leaf_organs:
        st = int(organ.getParameter("subType"))
        pos = st - 2  # subType 2 → position 0
        if pos < 0 or pos >= N_POSITIONS:
            continue

        nodes = organ.getNodes()
        if len(nodes) < 2:
            continue

        lrp = organ.getLeafRandomParameter()
        width_blade = lrp.Width_blade
        if width_blade < 0.01:
            continue

        skeleton = np.array([[n.x, n.y, n.z] for n in nodes], dtype=np.float32)

        # Compute per-node widths from leaf geometry profile
        phi = np.array(lrp.leafGeometryPhi)
        x_profile = np.array(lrp.leafGeometryX)

        if len(phi) > 0 and len(x_profile) > 0:
            diffs = np.diff(skeleton, axis=0)
            seg_lens = np.linalg.norm(diffs, axis=1)
            cumulative = np.concatenate([[0.0], np.cumsum(seg_lens)])
            total_len = cumulative[-1]
            if total_len > 1e-12:
                fracs = cumulative / total_len
                phi_min, phi_max = phi.min(), phi.max()
                node_phi = phi_min + fracs * (phi_max - phi_min)
                rel_widths = np.interp(node_phi, phi, x_profile)
                widths = rel_widths * width_blade  # half-width
            else:
                widths = np.full(len(nodes), width_blade, dtype=np.float32)
        else:
            widths = np.full(len(nodes), width_blade, dtype=np.float32)

        # Truncate or pad to n_max
        n_nodes = min(len(skeleton), n_max)
        skeletons[pos, :n_nodes, :3] = skeleton[:n_nodes]
        skeletons[pos, :n_nodes, 3] = widths[:n_nodes]
        masks[pos, :n_nodes] = True
        n_leaves += 1

    return skeletons, masks, n_leaves


def generate_single(
    params: dict,
    day: int = 60,
    seed: int = 0,
    template_xml: str | None = None,
) -> dict | None:
    """Run CPlantBox with given params and extract skeletons.

    Args:
        params: Sampled parameter dict (position → {param_name: value}).
        day: Simulation days.
        seed: CPlantBox random seed.
        template_xml: Path to calibrated XML template. If None, uses
            the default from ``dart.coupling.config``.

    Returns:
        Dict with keys ``params`` (133,), ``skeletons`` (11, N_MAX, 4),
        ``masks`` (11, N_MAX), ``n_leaves`` (int).  Or ``None`` if the
        simulation fails.
    """
    if template_xml is None:
        from dart.coupling.config import DEFAULT_XML
        template_xml = str(DEFAULT_XML)

    try:
        # Write modified XML to a temp file
        with tempfile.NamedTemporaryFile(
            suffix=".xml", delete=False, mode="wb"
        ) as tmp:
            tmp_path = tmp.name

        _apply_params_to_xml(template_xml, params, tmp_path)

        # Grow the plant (import here to isolate CPlantBox from worker init)
        from dart.coupling.growth.grow import grow_plant

        # Redirect CPlantBox stdout to avoid flooding logs
        devnull = open(os.devnull, "w")
        import sys
        old_stdout = sys.stdout
        sys.stdout = devnull

        try:
            plant = grow_plant(
                xml_path=tmp_path,
                simulation_time=day,
                min_stem_nodes=10,
                min_leaf_nodes=10,
                enable_photosynthesis=False,
                seed=seed,
            )
        finally:
            sys.stdout = old_stdout
            devnull.close()

        skeletons, masks, n_leaves = _extract_leaf_skeletons(plant)
        flat_params = _flatten_params(params)

        return {
            "params": flat_params,
            "skeletons": skeletons,
            "masks": masks,
            "n_leaves": n_leaves,
        }

    except Exception:
        logger.warning("generate_single failed:\n%s", traceback.format_exc())
        return None

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _worker_fn(args: tuple) -> dict | None:
    """Worker function for multiprocessing pool."""
    params, day, seed, template_xml = args
    return generate_single(params, day=day, seed=seed, template_xml=template_xml)


def generate_dataset(
    n_samples: int,
    output_path: str,
    day: int = 60,
    n_workers: int = 4,
    seed: int = 42,
    stats_path: str | None = None,
    template_xml: str | None = None,
) -> None:
    """Generate training dataset and save as HDF5.

    Samples parameters via Latin Hypercube, runs CPlantBox for each
    sample in parallel, and writes results to an HDF5 file with
    datasets: ``params`` (N, 133), ``skeletons`` (N, 11, 64, 4),
    ``masks`` (N, 11, 64).

    Args:
        n_samples: Number of samples to generate.
        output_path: Path to output HDF5 file.
        day: Growth simulation days.
        n_workers: Number of parallel CPlantBox processes.
        seed: Base random seed.
        stats_path: Path to ``maizefield3d_stats.json``.  If None, uses
            the default from ``dart.coupling.config``.
        template_xml: Path to calibrated XML.  If None, uses default.
    """
    if stats_path is None:
        from dart.coupling.config import DATA_DIR
        stats_path = str(DATA_DIR / "maizefield3d_stats.json")
    if template_xml is None:
        from dart.coupling.config import DEFAULT_XML
        template_xml = str(DEFAULT_XML)

    bounds = load_prior_bounds(stats_path)
    samples = sample_params(n_samples, bounds, seed=seed)

    # Build worker arguments: each sample gets a unique CPlantBox seed
    args_list = [
        (s, day, seed + i, template_xml)
        for i, s in enumerate(samples)
    ]

    logger.info(
        "Generating %d samples (day=%d, workers=%d) ...", n_samples, day, n_workers
    )

    results: list[dict] = []

    if n_workers <= 1:
        for i, args in enumerate(args_list):
            result = _worker_fn(args)
            if result is not None:
                results.append(result)
            if (i + 1) % 50 == 0 or i == len(args_list) - 1:
                logger.info("  %d / %d done (%d valid)", i + 1, n_samples, len(results))
    else:
        with mp.Pool(n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_worker_fn, args_list)):
                if result is not None:
                    results.append(result)
                if (i + 1) % 50 == 0 or i == len(args_list) - 1:
                    logger.info(
                        "  %d / %d done (%d valid)", i + 1, n_samples, len(results)
                    )

    if not results:
        raise RuntimeError("All samples failed — check CPlantBox configuration.")

    n_valid = len(results)
    logger.info("Writing %d valid samples to %s", n_valid, output_path)

    # Stack arrays
    all_params = np.stack([r["params"] for r in results])  # (N, 133)
    all_skeletons = np.stack([r["skeletons"] for r in results])  # (N, 11, 64, 4)
    all_masks = np.stack([r["masks"] for r in results])  # (N, 11, 64)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("params", data=all_params, compression="gzip")
        f.create_dataset("skeletons", data=all_skeletons, compression="gzip")
        f.create_dataset("masks", data=all_masks, compression="gzip")
        f.attrs["n_samples"] = n_valid
        f.attrs["n_requested"] = n_samples
        f.attrs["day"] = day
        f.attrs["seed"] = seed
        f.attrs["n_positions"] = N_POSITIONS
        f.attrs["n_max_nodes"] = N_MAX
        f.attrs["n_params"] = N_PARAMS

    logger.info(
        "Done. %d / %d samples valid (%.1f%%)",
        n_valid, n_samples, 100.0 * n_valid / n_samples,
    )
