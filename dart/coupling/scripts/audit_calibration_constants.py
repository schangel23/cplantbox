#!/usr/bin/env python3
"""Inventory every literature-derived calibration constant in the repo.

Walks three sources:

  1. ``dart/coupling/data/*.json`` — the JSON master files, including
     per-rank arrays and ``_meta`` provenance blocks.
  2. ``src/structural/*.{cpp,h}`` — C++ ``constexpr`` / ``static const``
     literals with adjacent comment-style citations.
  3. ``dart/coupling/**/*.py`` — module-level ``UPPER_CASE = NUMBER``
     assignments with adjacent comment-style citations.

Emits a single XML report with sections for each source plus a flagged
``<duplicates>`` block listing constants that appear in more than one
home (the "three places per number" footgun the audit flagged).

Run from the CPlantBox repo root:

    cpbenv/bin/python3 -m dart.coupling.scripts.audit_calibration_constants
        [--output PATH | --stdout]

By default writes to ``dart/coupling/data/calibration_inventory.xml``
and prints a one-line summary to stdout. Pass ``--stdout`` to dump the
full XML to stdout instead.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET
from xml.dom import minidom

# Regex helpers — kept narrow on purpose.
RE_CPP_CONSTEXPR = re.compile(
    r"^\s*(?:constexpr|static\s+const)\s+(?:double|float|int)\s+"
    r"([A-Z_][A-Z0-9_]*)\s*=\s*([^;]+);\s*(?://\s*(.*))?$"
)
RE_PY_MODULE_CONST = re.compile(
    r"^([A-Z_][A-Z0-9_]+)\s*=\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)"
    r"\s*(?:#\s*(.*))?$"
)
# Citation heuristic: any 4-digit year, or known author surname / paper
# tag, in the trailing comment marks the constant as literature-derived.
RE_CITATION = re.compile(
    r"\b(?:1[89]\d{2}|20\d{2}|"
    r"FA\s?\d{4}|AHB|Andrieu|Vidal|Padilla|Birch|Hesketh|Hillier|"
    r"Tardieu|Reymond|Fournier|Nielsen|Zhu|Couvreur|Tuzet|"
    r"Tollenaar|Warrington|Kanemasu|"
    r"plan\s*§)",
    re.IGNORECASE,
)

REPO = Path(__file__).resolve().parents[3]
DATA_DIR = REPO / "dart" / "coupling" / "data"
SRC_DIR = REPO / "src" / "structural"
PY_DIR = REPO / "dart" / "coupling"
DEFAULT_OUTPUT = DATA_DIR / "calibration_inventory.xml"


def head_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# JSON walker
# ---------------------------------------------------------------------------

def walk_json_numbers(obj, prefix: str = "") -> Iterable[tuple[str, float | int]]:
    """Yield (dotted_path, value) for every number in a nested dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_json_numbers(v, f"{prefix}{k}." if prefix else f"{k}.")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_json_numbers(v, f"{prefix.rstrip('.')}[{i}].")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix.rstrip("."), obj


def collect_json_provenance(obj) -> dict[str, str]:
    """Pull citation-like fields from a `_meta` block (if present)."""
    if not isinstance(obj, dict):
        return {}
    meta = obj.get("_meta") or obj.get("meta")
    if not isinstance(meta, dict):
        # Fall back to top-level _description / source / cultivar keys.
        out = {}
        for k in ("source", "_description", "_source_notes", "cultivar"):
            if k in obj and isinstance(obj[k], str):
                out[k] = obj[k]
        return out
    out = {}
    for k, v in meta.items():
        if isinstance(v, str) and (RE_CITATION.search(v) or k.startswith("source")
                                   or "cultivar" in k or "note" in k):
            out[k] = v
    return out


def scan_jsons(root: ET.Element) -> int:
    section = ET.SubElement(root, "json_files")
    n_files = 0
    for path in sorted(DATA_DIR.glob("*.json")):
        if path.name == DEFAULT_OUTPUT.name:
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        n_files += 1
        f = ET.SubElement(section, "file", path=str(path.relative_to(REPO)))
        prov = collect_json_provenance(data)
        for k, v in prov.items():
            ET.SubElement(f, "provenance", key=k).text = v[:600]
        n_constants = 0
        for path_dotted, value in walk_json_numbers(data):
            ET.SubElement(
                f, "constant",
                path=path_dotted,
                value=repr(value),
            )
            n_constants += 1
        f.set("n_constants", str(n_constants))
    section.set("n_files", str(n_files))
    return n_files


# ---------------------------------------------------------------------------
# C++ walker
# ---------------------------------------------------------------------------

def scan_cpp(root: ET.Element) -> list[dict]:
    section = ET.SubElement(root, "cpp_constants")
    found = []
    for path in sorted(SRC_DIR.glob("*.cpp")) + sorted(SRC_DIR.glob("*.h")):
        rel = str(path.relative_to(REPO))
        with path.open() as fh:
            for lineno, line in enumerate(fh, 1):
                m = RE_CPP_CONSTEXPR.match(line)
                if not m:
                    continue
                name, value, comment = m.group(1), m.group(2).strip(), (m.group(3) or "").strip()
                # Skip non-numeric (e.g. enum-class or string literals).
                value_clean = value.split("//")[0].strip().rstrip(";")
                if not re.match(r"^-?[\d\.eE+\-]+$", value_clean):
                    continue
                attrs = {
                    "name": name,
                    "value": value_clean,
                    "file": rel,
                    "line": str(lineno),
                }
                if comment:
                    attrs["citation"] = comment
                ET.SubElement(section, "constant", **attrs)
                found.append(attrs)
    section.set("n_constants", str(len(found)))
    return found


# ---------------------------------------------------------------------------
# Python walker
# ---------------------------------------------------------------------------

PY_SKIP_DIRS = {"tests", "__pycache__", "output", "build", "cpbenv"}


def scan_python(root: ET.Element) -> list[dict]:
    section = ET.SubElement(root, "python_constants")
    found = []
    for path in sorted(PY_DIR.rglob("*.py")):
        rel_parts = path.relative_to(REPO).parts
        if any(p in PY_SKIP_DIRS for p in rel_parts):
            continue
        rel = str(path.relative_to(REPO))
        try:
            lines = path.read_text().splitlines()
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(lines, 1):
            m = RE_PY_MODULE_CONST.match(line)
            if not m:
                continue
            name, value, comment = m.group(1), m.group(2), (m.group(3) or "").strip()
            attrs = {
                "name": name,
                "value": value,
                "file": rel,
                "line": str(lineno),
            }
            if comment:
                attrs["citation"] = comment
            ET.SubElement(section, "constant", **attrs)
            found.append(attrs)
    section.set("n_constants", str(len(found)))
    return found


# ---------------------------------------------------------------------------
# Duplicate detector
# ---------------------------------------------------------------------------

def find_duplicates(
    cpp: list[dict],
    py: list[dict],
    root: ET.Element,
) -> int:
    """Group constants by (name, value) across C++ and Python homes."""
    by_key: dict[tuple[str, str], list[dict]] = {}
    for entry in cpp + py:
        key = (entry["name"], entry["value"])
        by_key.setdefault(key, []).append(entry)
    section = ET.SubElement(root, "duplicates")
    n_dups = 0
    for (name, value), homes in sorted(by_key.items()):
        if len(homes) < 2:
            continue
        n_dups += 1
        d = ET.SubElement(section, "duplicate", name=name, value=value,
                          n_homes=str(len(homes)))
        for h in homes:
            attrs = {"file": h["file"], "line": h["line"]}
            if "citation" in h:
                attrs["citation"] = h["citation"]
            ET.SubElement(d, "location", **attrs)
    section.set("n_duplicates", str(n_dups))
    return n_dups


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def pretty(root: ET.Element) -> str:
    raw = ET.tostring(root, encoding="unicode")
    return minidom.parseString(raw).toprettyxml(indent="  ")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="audit_calibration_constants",
        description="Walk JSON / C++ / Python and emit one XML inventory.",
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"output XML path (default: {DEFAULT_OUTPUT})")
    p.add_argument("--stdout", action="store_true",
                   help="print full XML to stdout instead of writing a file")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = ET.Element("calibration_inventory", attrib={
        "generated": date.today().isoformat(),
        "head_commit": head_commit(),
        "repo": str(REPO),
    })
    n_json = scan_jsons(root)
    cpp = scan_cpp(root)
    py = scan_python(root)
    n_dups = find_duplicates(cpp, py, root)

    output = pretty(root)

    header = (
        "<!-- Generated by dart/coupling/scripts/audit_calibration_constants.py.\n"
        "     Do not edit by hand — regenerate after touching any calibration\n"
        "     constant via:\n"
        "         cpbenv/bin/python3 -m dart.coupling.scripts.audit_calibration_constants\n"
        "     Sections: <json_files>, <cpp_constants>, <python_constants>, <duplicates>.\n"
        "     The <duplicates> block flags constants that exist in more than one home;\n"
        "     a non-empty list is the consolidation backlog. -->\n"
    )
    # Insert the header right after the XML declaration line.
    output_lines = output.splitlines(keepends=True)
    output = output_lines[0] + header + "".join(output_lines[1:])

    if args.stdout:
        sys.stdout.write(output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)

    # One-line summary always goes to stderr so it's visible even when piped.
    print(
        f"calibration_inventory: {n_json} JSON files, "
        f"{len(cpp)} C++ literals, {len(py)} Python literals, "
        f"{n_dups} duplicate (name,value) groups"
        + (f" → {args.output.relative_to(REPO)}" if not args.stdout else ""),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
