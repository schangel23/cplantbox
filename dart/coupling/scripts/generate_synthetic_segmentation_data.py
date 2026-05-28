"""Generate a batch of labelled synthetic maize point clouds for sim-to-real
leaf instance + rank segmentation training.

Each output cloud carries per-point organ_id (instance), organ_type
(stem/leaf/midrib/tassel), and rank (leaf base z order). Saved as LAS + NPZ
to keep CloudCompare and pure-numpy pipelines both happy.

Example:
    python -m dart.coupling.scripts.generate_synthetic_segmentation_data \\
        --xml dart/coupling/data/maize_calibrated.xml \\
        --out dart/coupling/output/synthetic_segdata \\
        --n-plants 1 --sim-time 60 --seeds 42
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", required=True, help="CPlantBox XML (e.g. maize_calibrated.xml)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n-plants", type=int, default=1)
    ap.add_argument("--sim-time", type=int, default=60, help="growth days per plant")
    ap.add_argument("--seeds", type=int, nargs="*", help="explicit seeds (else 1..n)")
    ap.add_argument("--n-points", type=int, default=120_000, help="points per plant")
    ap.add_argument("--noise-mm", type=float, default=0.5, help="Gaussian noise sigma (mm)")
    ap.add_argument("--no-las", action="store_true", help="skip LAS export")
    ap.add_argument("--no-npz", action="store_true", help="skip NPZ export")
    args = ap.parse_args(argv)

    from dart.coupling.geometry.synthetic_pointcloud import (
        generate_synthetic_pointcloud, save_las, save_npz)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds if args.seeds else list(range(1, args.n_plants + 1))
    if len(seeds) != args.n_plants:
        sys.exit(f"--seeds gave {len(seeds)} values but --n-plants={args.n_plants}")

    manifest = []
    for k, seed in enumerate(seeds):
        tag = f"plant_seed{seed:04d}_day{args.sim_time:03d}"
        print(f"[{k+1}/{len(seeds)}] {tag} ...")
        data = generate_synthetic_pointcloud(
            xml_path=args.xml, simulation_time=args.sim_time, seed=seed,
            n_points=args.n_points, noise_sigma_m=args.noise_mm / 1000.0,
        )
        meta = data["meta"]
        print(f"   leaves={meta['n_leaves']}  surface={meta['total_surface_area_m2']*1e4:.0f} cm^2  "
              f"points={meta['n_points']:,}")
        types, counts = np.unique(data["organ_type"], return_counts=True)
        print(f"   organ_type counts: {dict(zip(types.tolist(), counts.tolist()))}")
        rank_set = sorted(set(int(r) for r in data["rank"] if r > 0))
        print(f"   leaf ranks present: {rank_set}")
        if not args.no_npz:
            save_npz(out / f"{tag}.npz", data)
        if not args.no_las:
            save_las(out / f"{tag}.las", data)
        manifest.append({"file": tag, **meta})

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nWrote {len(manifest)} plants -> {out}")


if __name__ == "__main__":
    main()
