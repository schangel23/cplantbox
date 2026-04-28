#!/usr/bin/env bash
# Reproducible PlantGL build for CPlantBox cpbenv (Python 3.14, Qt6, Boost 1.90).
#
# Clones openalea/plantgl, applies the 4 patches in patches/, builds against
# the active venv, and copies the resulting openalea/plantgl/ tree into the
# venv's site-packages.
#
# Prereqs (Arch / pacman):
#   sudo pacman -S boost boost-libs cgal eigen qt6-base qt6-5compat cmake
#
# Usage:
#   source /path/to/cpbenv/bin/activate
#   bash dart/external/plantgl/build.sh

set -euo pipefail

CPBENV="${VIRTUAL_ENV:-}"
if [[ -z "$CPBENV" ]]; then
    echo "ERROR: activate cpbenv first (source cpbenv/bin/activate)" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_DIR="$SCRIPT_DIR/patches"
SRC_DIR="${PLANTGL_SRC:-/tmp/plantgl_src}"
UPSTREAM="https://github.com/openalea/plantgl.git"
SITE_PACKAGES="$CPBENV/lib/python$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages"

echo "==> Cloning openalea/plantgl into $SRC_DIR (depth 1)"
rm -rf "$SRC_DIR"
git clone --depth 1 "$UPSTREAM" "$SRC_DIR"

echo "==> Applying $(ls "$PATCHES_DIR"/*.patch | wc -l) patches"
cd "$SRC_DIR"
for p in "$PATCHES_DIR"/*.patch; do
    echo "    $(basename "$p")"
    git apply "$p"
done

echo "==> Configuring CMake (headless, Qt offscreen)"
mkdir -p build
cd build
QT_QPA_PLATFORM=offscreen cmake .. -DPython3_EXECUTABLE="$CPBENV/bin/python"

echo "==> Building (parallel)"
QT_QPA_PLATFORM=offscreen make -j"$(nproc)"

echo "==> Installing (in-tree; ignore the /share/plantgl error — that's the install_share patch we deferred)"
QT_QPA_PLATFORM=offscreen make install || true

echo "==> Copying openalea/plantgl into $SITE_PACKAGES"
mkdir -p "$SITE_PACKAGES/openalea"
cp -r "$SRC_DIR/src/openalea/plantgl" "$SITE_PACKAGES/openalea/"

echo "==> Patching site-packages all.py to make gui import optional"
ALL_PY="$SITE_PACKAGES/openalea/plantgl/all.py"
if ! grep -q "Headless-tolerant" "$ALL_PY"; then
    python - <<PY
from pathlib import Path
p = Path("$ALL_PY")
src = p.read_text()
old = "if not pgl_support_extension('PGL_NO_QT_GUI'):\n    from .gui import *"
new = """if not pgl_support_extension('PGL_NO_QT_GUI'):
    # Headless-tolerant: gui requires QT_API env var + a Qt python binding.
    # If neither is set up, skip the gui import — non-gui consumers (cpbenv
    # pipeline, openalea.mtg.plantframe.turtle) keep working.
    try:
        from .gui import *
    except (KeyError, ImportError):
        pass"""
p.write_text(src.replace(old, new))
PY
fi

echo "==> Smoke test"
python -c "from openalea.plantgl.scenegraph import NurbsPatch, Point4Matrix, RealArray; print('PlantGL OK', NurbsPatch.__name__)"
python -c "from openalea.plantgl.all import *; print('all-import OK (gui guarded)')"

echo "==> Done. PlantGL installed in $SITE_PACKAGES/openalea/plantgl/"
