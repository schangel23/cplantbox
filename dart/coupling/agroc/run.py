"""AgroC Fortran runner: prepare workdir, execute binary, validate outputs."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

# Default AgroC source directories (local / server)
_DEFAULT_AGROC_SRC_LOCAL = "/home/lukas/PHD/agroC_20250327_1511/src"
_DEFAULT_AGROC_SRC_SERVER = "/media/data/Lukas/agroC_20250327_1511/src"

# Required input files (besides selector.in which is always copied)
_REQUIRED_INPUTS = ["selector.in", "atmosph.in", "plants.in"]
_OPTIONAL_INPUTS = ["rrd.in"]


def get_agroc_src() -> Path:
    """Resolve AgroC source directory from AGROC_SRC env var or defaults."""
    env = os.environ.get("AGROC_SRC")
    if env:
        return Path(env)
    local = Path(_DEFAULT_AGROC_SRC_LOCAL)
    if local.exists():
        return local
    server = Path(_DEFAULT_AGROC_SRC_SERVER)
    if server.exists():
        return server
    raise FileNotFoundError(
        f"AgroC source not found. Set AGROC_SRC env var or check paths:\n"
        f"  Local:  {_DEFAULT_AGROC_SRC_LOCAL}\n"
        f"  Server: {_DEFAULT_AGROC_SRC_SERVER}"
    )


def enable_external_plant_mode(selector_path: Path) -> None:
    """Flip ExternalPlantMode flag from 'f' to 't' in selector.in.

    The flags line (line 9 in standard selector.in) has 11 space-separated
    boolean flags.  The last one is ExternalPlant.  We replace the last 'f'
    on that line with 't'.

    Also verifies PlantsExist (7th flag) is 't' — required by AgroC.
    """
    text = selector_path.read_text()
    lines = text.split("\n")

    # Find the line after the header containing "ExternalPlant"
    header_idx = None
    for i, line in enumerate(lines):
        if "ExternalPlant" in line and "ShortO" in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(
            f"Cannot find 'ShortO ... ExternalPlant' header in {selector_path}"
        )

    values_idx = header_idx + 1
    vals_line = lines[values_idx]

    # Parse boolean flags (t/f tokens)
    tokens = vals_line.split()
    if len(tokens) < 11:
        raise ValueError(
            f"Expected 11 flags on line {values_idx + 1}, got {len(tokens)}: {vals_line}"
        )

    # Check PlantsExist (index 6, 0-based) is 't'
    if tokens[6].lower() != "t":
        print(f"  WARNING: PlantsExist was '{tokens[6]}', setting to 't' "
              f"(required by ExternalPlantMode)")
        tokens[6] = "t"

    # Set ExternalPlantMode (index 10, last flag) to 't'
    tokens[10] = "t"

    # Reconstruct with aligned spacing to match original formatting
    # Use fixed-width formatting matching the original
    lines[values_idx] = "   " + "      ".join(tokens[:6]) + "        " + \
        "         ".join(tokens[6:8]) + "         " + \
        "        ".join(tokens[8:10]) + "          " + tokens[10]

    selector_path.write_text("\n".join(lines))


def enable_output_flags(selector_path: Path) -> None:
    """Flip t_level and nod_prod output flags from 'f' to 't' in selector.in.

    The output flags line (line 5 in standard selector.in) has 14 space-separated
    boolean flags:
      i_check run_inf t_level a_level co2_inf nod_inf balance point
      nod_pool reduction nod_prod matlab invers term

    t_level (index 2) and nod_prod (index 10) are needed for validation.
    """
    text = selector_path.read_text()
    lines = text.split("\n")

    # Find the header line with output flag names
    header_idx = None
    for i, line in enumerate(lines):
        if "t_level" in line and "nod_prod" in line and "i_check" in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(
            f"Cannot find 'i_check ... t_level ... nod_prod' header in {selector_path}"
        )

    values_idx = header_idx + 1
    tokens = lines[values_idx].split()
    if len(tokens) < 14:
        raise ValueError(
            f"Expected 14 output flags on line {values_idx + 1}, got {len(tokens)}"
        )

    # t_level = index 2, nod_prod = index 10
    changed = []
    if tokens[2].lower() != "t":
        tokens[2] = "t"
        changed.append("t_level")
    if tokens[10].lower() != "t":
        tokens[10] = "t"
        changed.append("nod_prod")

    if changed:
        # Reconstruct with consistent spacing
        lines[values_idx] = "  " + "      ".join(tokens)
        selector_path.write_text("\n".join(lines))


def _read_coupling_time_range(coupling_csv_path: Path) -> tuple:
    """Read the coupling CSV and return (day_min, day_max) from time_d column."""
    text = coupling_csv_path.read_text().strip()
    lines = text.split("\n")
    header = lines[0].split(",")
    time_idx = header.index("time_d")
    days = [float(line.split(",")[time_idx]) for line in lines[1:] if line.strip()]
    return min(days), max(days)


def trim_atmosph_to_range(atmosph_path: Path, day_max: float) -> None:
    """Truncate atmosph.in to keep only rows up to day_max.

    atmosph.in format: 12 header lines, then daily data rows (first column = day),
    then 'end' line and footer.
    """
    text = atmosph_path.read_text()
    lines = text.split("\n")

    # Split into header (lines starting with * or containing tInit), data, and footer
    header_lines = []
    data_lines = []
    in_data = False
    for line in lines:
        stripped = line.strip()
        if not in_data:
            header_lines.append(line)
            # Data starts after the units line (line 12 in standard format)
            # Detect transition: first line whose first token is a number
            if stripped and not stripped.startswith("*") and stripped[0].isdigit():
                # This is actually a data line, move it
                header_lines.pop()
                data_lines.append(line)
                in_data = True
        else:
            if stripped.lower() == "end" or stripped.startswith("***"):
                break
            data_lines.append(line)

    # Keep data rows where day <= day_max (with 1-day buffer)
    keep_limit = int(day_max) + 1
    kept = []
    for line in data_lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            day = int(stripped.split()[0])
            if day <= keep_limit:
                kept.append(line)
        except (ValueError, IndexError):
            continue

    # Reconstruct
    result = "\n".join(header_lines) + "\n"
    result += "\n".join(kept) + "\n"
    result += "end  \n"
    result += "*** END OF INPUT FILE 'ATMOSPH.IN' *************************************\n"
    atmosph_path.write_text(result)


def trim_selector_tprint(selector_path: Path, day_min: float,
                         day_max: float) -> None:
    """Update TPrint in selector.in to match the coupling time range.

    Sets NumOfPrints and TPrint values to cover the coupling range with
    reasonable spacing.
    """
    text = selector_path.read_text()
    lines = text.split("\n")

    # Generate print times: day_min, day_max, and evenly spaced in between
    d_min = int(day_min)
    d_max = int(day_max)
    if d_max - d_min <= 10:
        tprints = list(range(d_min, d_max + 1))
    else:
        # Every 5 days, plus endpoints
        tprints = list(range(d_min, d_max + 1, 5))
        if tprints[-1] != d_max:
            tprints.append(d_max)

    # Find the NumOfPrints line (contains "NumOfPrints")
    nop_idx = None
    for i, line in enumerate(lines):
        if "NumOfPrints" in line:
            nop_idx = i
            break

    if nop_idx is None:
        return  # Can't find it, skip silently

    # The values line is nop_idx + 1, TPrint header is nop_idx + 2, TPrint values is nop_idx + 3
    vals_line = lines[nop_idx + 1].split()
    # NumOfPrints is the last token on the values line
    vals_line[-1] = str(len(tprints))
    lines[nop_idx + 1] = "   ".join(vals_line)

    # TPrint values line (nop_idx + 3)
    lines[nop_idx + 3] = " ".join(str(t) for t in tprints)

    selector_path.write_text("\n".join(lines))


def prepare_agroc_workdir(agroc_src: Path, output_dir: Path,
                          coupling_csv_path: Path) -> Path:
    """Copy AgroC binary + inputs to a working directory, enable ExternalPlantMode.

    Args:
        agroc_src: Path to AgroC source directory containing binary and .in files.
        output_dir: Directory to create the AgroC working directory in.
        coupling_csv_path: Path to coupling.csv from Step 5.

    Returns:
        Path to the prepared working directory.
    """
    workdir = output_dir / "agroc_run"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    # Copy binary
    binary_src = agroc_src / "agroC"
    if not binary_src.exists():
        raise FileNotFoundError(f"AgroC binary not found at {binary_src}")
    shutil.copy2(binary_src, workdir / "agroC")
    os.chmod(workdir / "agroC", 0o755)

    # Copy required input files
    for fname in _REQUIRED_INPUTS:
        src = agroc_src / fname
        if not src.exists():
            raise FileNotFoundError(f"Required AgroC input not found: {src}")
        shutil.copy2(src, workdir / fname)

    # Copy optional input files
    for fname in _OPTIONAL_INPUTS:
        src = agroc_src / fname
        if src.exists():
            shutil.copy2(src, workdir / fname)

    # Copy coupling CSV (must be named exactly "coupling.csv" for Fortran)
    if not coupling_csv_path.exists():
        raise FileNotFoundError(f"Coupling CSV not found: {coupling_csv_path}")
    shutil.copy2(coupling_csv_path, workdir / "coupling.csv")

    # Enable ExternalPlantMode in selector.in
    enable_external_plant_mode(workdir / "selector.in")
    print(f"  Enabled ExternalPlantMode in {workdir / 'selector.in'}")

    # Enable t_level and nod_prod output flags
    enable_output_flags(workdir / "selector.in")
    print(f"  Enabled t_level + nod_prod output flags")

    # Trim simulation to coupling CSV time range
    try:
        day_min, day_max = _read_coupling_time_range(coupling_csv_path)
        trim_atmosph_to_range(workdir / "atmosph.in", day_max)
        trim_selector_tprint(workdir / "selector.in", day_min, day_max)
        print(f"  Trimmed simulation to days {int(day_min)}-{int(day_max)}")
    except Exception as e:
        print(f"  WARNING: could not trim to coupling range: {e}")

    return workdir


def run_agroc(workdir: Path, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run the AgroC binary in the prepared working directory.

    Args:
        workdir: Working directory with agroC binary and input files.
        timeout: Max seconds to wait (default 300 = 5 min).

    Returns:
        CompletedProcess with stdout/stderr captured.
    """
    binary = workdir / "agroC"
    if not binary.exists():
        raise FileNotFoundError(f"AgroC binary not found at {binary}")

    print(f"  Running AgroC in {workdir}...")
    proc = subprocess.run(
        ["./agroC"],
        cwd=str(workdir),
        capture_output=True, text=True,
        timeout=timeout,
    )

    if proc.returncode != 0:
        print(f"  AgroC exit code: {proc.returncode}")
        if proc.stderr:
            print(f"  stderr (last 500 chars): {proc.stderr[-500:]}")
    else:
        print(f"  AgroC completed successfully (exit code 0)")

    return proc


def parse_t_level(path: Path) -> dict:
    """Parse t_level.out and extract key columns.

    Returns dict with column arrays. Common columns:
      Time, rTop, rRoot, vTop, vRoot, vBot, sum(rTop), Runoff,
      GPP, NPP, aboveground_respiration, root_respiration
    """
    if not path.exists():
        return {"exists": False}

    text = path.read_text().strip()
    lines = text.split("\n")
    if len(lines) < 2:
        return {"exists": True, "n_rows": 0}

    # Find header line containing column names (skip banner and units lines)
    header_line = None
    data_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith(("!", "*")):
            continue
        # Skip units lines like "[T]  [-]  [L/T] ..."
        if stripped.startswith("["):
            continue
        # The header line contains "Time" and column names
        if "Time" in stripped and not stripped[0].isdigit():
            header_line = stripped
            # Skip the units line that follows the header
            data_start = i + 1
            if data_start < len(lines) and lines[data_start].strip().startswith("["):
                data_start += 1
            break

    if header_line is None:
        return {"exists": True, "n_rows": 0}

    headers = header_line.split()
    result = {"exists": True, "headers": headers, "columns": {}}

    # Parse data rows
    data_rows = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            vals = [float(v) for v in stripped.split()]
            if len(vals) == len(headers):
                data_rows.append(vals)
        except ValueError:
            continue

    result["n_rows"] = len(data_rows)

    if data_rows:
        data = np.array(data_rows)
        for j, h in enumerate(headers):
            result["columns"][h] = data[:, j].tolist()

    return result


def parse_nod_inf(path: Path) -> dict:
    """Parse nod_inf.out — per-node soil state at each print time.

    File format (Fortran fixed-width, whitespace-delimited):
      3 banner lines (* header)
      2 column header lines (#Node ... / #  [L] ...)
      Repeating blocks:
        #Time:  <day>
        218 data rows (node N→1, deepest→shallowest)

    Returns:
        dict with:
            exists: bool
            times: list of print-time days
            columns: ['Node','Depth','Head','Moisture','CO2','Temp',
                       'K','C','Flux','Sink','Product']
            snapshots: {day: np.ndarray shape (n_nodes, n_cols)}
    """
    if not path.exists():
        return {"exists": False}

    text = path.read_text().strip()
    lines = text.split("\n")

    # Column names from the #Node header line
    col_names = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#Node"):
            col_names = stripped.lstrip("#").split()
            break

    if col_names is None:
        return {"exists": True, "times": [], "snapshots": {}}

    # Parse time blocks
    times = []
    snapshots = {}
    current_time = None
    current_rows = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#Time:"):
            # Flush previous block
            if current_time is not None and current_rows:
                snapshots[current_time] = np.array(current_rows)
                times.append(current_time)
            current_time = int(stripped.split(":")[1])
            current_rows = []
            continue
        if current_time is None:
            continue
        if stripped.startswith("#") or stripped.startswith("*") or not stripped:
            continue
        try:
            vals = [float(v) for v in stripped.split()[:len(col_names)]]
            if len(vals) == len(col_names):
                current_rows.append(vals)
        except ValueError:
            continue

    # Flush last block
    if current_time is not None and current_rows:
        snapshots[current_time] = np.array(current_rows)
        times.append(current_time)

    return {
        "exists": True,
        "times": times,
        "columns": col_names,
        "snapshots": snapshots,
    }


def extract_soil_state(nod_inf: dict, day: int = None) -> dict:
    """Extract soil temperature and moisture profiles from parsed nod_inf.

    Provides the same data that interface_output.csv would have contained.

    Args:
        nod_inf: dict from parse_nod_inf().
        day: specific print day to extract (default: last available).

    Returns:
        dict with depth_cm, temperature_C, moisture arrays (surface→deep),
        or None if data not available.
    """
    if not nod_inf.get("exists") or not nod_inf.get("snapshots"):
        return None

    if day is None:
        day = nod_inf["times"][-1]

    if day not in nod_inf["snapshots"]:
        return None

    cols = nod_inf["columns"]
    data = nod_inf["snapshots"][day]

    # Column indices
    try:
        i_depth = cols.index("Depth")
        i_temp = cols.index("Temp")
        i_moist = cols.index("Moisture")
        i_head = cols.index("Head")
    except ValueError:
        return None

    # Data is stored deepest→shallowest (node N→1), reverse to surface→deep
    depth = data[:, i_depth][::-1]
    temp = data[:, i_temp][::-1]
    moisture = data[:, i_moist][::-1]
    head = data[:, i_head][::-1]

    return {
        "day": day,
        "depth_cm": depth.tolist(),
        "temperature_C": temp.tolist(),
        "moisture": moisture.tolist(),
        "head_cm": head.tolist(),
        "n_nodes": len(depth),
    }


def validate_agroc_outputs(workdir: Path, coupling_csv_path: Path = None) -> dict:
    """Check AgroC outputs exist and validate key metrics.

    Checks:
      1. t_level.out exists and has data rows
      2. nod_prod.out exists (root production/respiration)
      3. If coupling_csv provided, compare GPP from t_level vs CSV input

    Returns:
        dict with validation results.
    """
    result = {"passed": True, "checks": []}

    # Check t_level.out (fall back to a_level.out for older configs)
    t_level_path = workdir / "t_level.out"
    a_level_path = workdir / "a_level.out"
    t_level = parse_t_level(t_level_path)

    if not t_level.get("exists", False):
        # Try a_level.out as fallback
        t_level = parse_t_level(a_level_path)
        if t_level.get("exists", False) and t_level.get("n_rows", 0) > 0:
            result["checks"].append(
                f"OK: a_level.out has {t_level['n_rows']} rows "
                f"(t_level.out not found, using fallback), "
                f"columns: {t_level['headers']}"
            )
            result["t_level"] = t_level
        else:
            result["checks"].append(
                "FAIL: neither t_level.out nor a_level.out found"
            )
            result["passed"] = False
    elif t_level.get("n_rows", 0) == 0:
        result["checks"].append("FAIL: t_level.out has no data rows")
        result["passed"] = False
    else:
        result["checks"].append(
            f"OK: t_level.out has {t_level['n_rows']} rows, "
            f"columns: {t_level['headers']}"
        )
        result["t_level"] = t_level

    # Check nod_prod.out
    nod_prod_path = workdir / "nod_prod.out"
    if nod_prod_path.exists() and nod_prod_path.stat().st_size > 0:
        result["checks"].append(
            f"OK: nod_prod.out exists ({nod_prod_path.stat().st_size} bytes)"
        )
    else:
        result["checks"].append("WARN: nod_prod.out not found or empty")

    # GPP comparison with coupling CSV
    if coupling_csv_path and coupling_csv_path.exists() and "columns" in t_level:
        gpp_col = t_level["columns"].get("GPP")
        if gpp_col is not None and len(gpp_col) > 0:
            # Read GPP from coupling CSV
            try:
                csv_text = coupling_csv_path.read_text().strip()
                csv_lines = csv_text.split("\n")
                csv_header = csv_lines[0].split(",")
                gpp_idx = csv_header.index("GPP_mol_co2_per_cm2_d")
                csv_gpp_values = []
                for line in csv_lines[1:]:
                    vals = line.split(",")
                    csv_gpp_values.append(float(vals[gpp_idx]))

                if csv_gpp_values:
                    csv_gpp_mean = np.mean(csv_gpp_values)
                    tlevel_gpp_mean = np.mean(gpp_col)

                    if csv_gpp_mean > 0:
                        rel_diff = abs(tlevel_gpp_mean - csv_gpp_mean) / csv_gpp_mean
                        result["gpp_csv_mean"] = csv_gpp_mean
                        result["gpp_tlevel_mean"] = tlevel_gpp_mean
                        result["gpp_rel_diff"] = rel_diff
                        if rel_diff <= 0.05:
                            result["checks"].append(
                                f"OK: GPP match — CSV={csv_gpp_mean:.6e}, "
                                f"t_level={tlevel_gpp_mean:.6e} (diff={rel_diff:.2%})"
                            )
                        else:
                            result["checks"].append(
                                f"WARN: GPP mismatch — CSV={csv_gpp_mean:.6e}, "
                                f"t_level={tlevel_gpp_mean:.6e} (diff={rel_diff:.2%})"
                            )
            except Exception as e:
                result["checks"].append(f"WARN: Could not compare GPP: {e}")

    # Check nod_inf.out — soil state (replaces broken interface_output.csv)
    nod_inf_path = workdir / "nod_inf.out"
    nod_inf = parse_nod_inf(nod_inf_path)
    if nod_inf.get("exists") and nod_inf.get("times"):
        soil = extract_soil_state(nod_inf)
        if soil is not None:
            result["checks"].append(
                f"OK: nod_inf.out has {len(nod_inf['times'])} snapshots "
                f"(days {nod_inf['times']}), "
                f"{soil['n_nodes']} nodes, "
                f"Tsoil {min(soil['temperature_C']):.1f}-"
                f"{max(soil['temperature_C']):.1f}°C"
            )
            result["soil_state"] = soil
            result["nod_inf"] = nod_inf
        else:
            result["checks"].append("WARN: nod_inf.out exists but could not extract soil state")
    else:
        result["checks"].append("WARN: nod_inf.out not found or empty")

    return result


# ---------------------------------------------------------------------------
# CLI entry point (called from __main__.py)
# ---------------------------------------------------------------------------

def main_agroc_run(args):
    """CLI handler for the ``agroc-run`` subcommand."""
    import sys
    from pathlib import Path as _Path
    from ..config import OUTPUT_DIR

    # Resolve agroc source
    if args.agroc_src:
        agroc_src = _Path(args.agroc_src)
    else:
        agroc_src = get_agroc_src()

    # Resolve output dir
    if args.output_dir:
        out_dir = _Path(args.output_dir)
    else:
        out_dir = OUTPUT_DIR / "agroc_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    coupling_csv = _Path(args.coupling_csv)

    print(f"\n{'='*60}")
    print("AGROC RUN — ExternalPlantMode")
    print(f"{'='*60}")
    print(f"  AgroC source:  {agroc_src}")
    print(f"  Coupling CSV:  {coupling_csv}")
    print(f"  Output:        {out_dir}")

    # 1. Prepare working directory
    workdir = prepare_agroc_workdir(agroc_src, out_dir, coupling_csv)
    print(f"  Working dir:   {workdir}")

    # 2. Run AgroC
    proc = run_agroc(workdir, timeout=args.timeout)

    if proc.returncode != 0:
        print(f"\n  AgroC FAILED (exit code {proc.returncode})")
        # Save stdout/stderr for debugging
        (workdir / "agroc_stdout.txt").write_text(proc.stdout or "")
        (workdir / "agroc_stderr.txt").write_text(proc.stderr or "")
        sys.exit(1)

    # 3. Validate outputs
    validation = validate_agroc_outputs(workdir, coupling_csv)
    print(f"\n  Validation:")
    for check in validation["checks"]:
        print(f"    {check}")

    if validation["passed"]:
        print(f"\n  AGROC RUN PASSED")
    else:
        print(f"\n  AGROC RUN — validation warnings (see above)")
