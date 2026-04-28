# PlantGL build recipe (cpbenv, Python 3.14, Qt6/Boost 1.90)

PlantGL (`openalea.plantgl`) is not on PyPI; it must be built from source against
system Qt/Boost/CGAL/Eigen. This directory captures the patches and build steps
needed to make it compile under the current Arch (Boost 1.90, Qt6, Python 3.14)
toolchain.

## Why these patches exist

| # | Patch | What | Why |
|---|---|---|---|
| 01 | `01-boost-system-removed.patch` | Drops `Boost::system` from CMake link lines (4 files) | Boost 1.90 made `system` header-only; the `Boost::system` CMake target no longer exists |
| 02 | `02-ann-guard-wireselection.patch` | Wraps `ThreadAdapterWireSelection::main` body in `#ifdef PGL_WITH_ANN` | Arch has no ANN library; the function uses `KDTree3` (an alias for `ANNKDTree3`) without a guard |
| 03 | `03-qt-no-keywords-pglalgo.patch` | Adds `target_compile_definitions(_pglalgo PRIVATE QT_NO_KEYWORDS)` | Qt's `slots` macro collides with Python 3.14's `PyType_Spec.slots` field name in the algo wrapper |
| 04 | `04-undef-slots-before-python.patch` | Adds `#undef slots` before `<boost/python.hpp>` in two PlantGL python helpers | Defence-in-depth so Python wrapper headers stay buildable even without target-level `QT_NO_KEYWORDS` |

The original install used 5 patches; the 5th (commenting out `install_share()`)
isn't applied here — instead `build.sh` ignores the install-share error since
the share files aren't needed for our pipeline. If you do need them, you can
add the original patch back.

## Building

```bash
# 1. ensure system deps are installed
sudo pacman -S boost boost-libs cgal eigen qt6-base qt6-5compat cmake

# 2. activate cpbenv (must already exist with Python 3.14)
source /path/to/CPlantBox/cpbenv/bin/activate

# 3. run the build script
bash dart/external/plantgl/build.sh
```

The script clones upstream `openalea/plantgl@master`, applies the 4 patches,
configures + builds with `QT_QPA_PLATFORM=offscreen`, and copies the resulting
`openalea/plantgl/` package into the active venv's `site-packages/`. It also
patches `site-packages/openalea/plantgl/all.py` to make the `gui` import
optional (avoids `KeyError: 'QT_API'` when no Qt python binding is installed).

End state: `from openalea.plantgl.scenegraph import NurbsPatch` works.

## When to re-run

- After `git submodule deinit` / venv recreation
- After upstream openalea/plantgl makes API changes that break our patches
  (then regenerate patches with `git diff` from `/tmp/plantgl_src` and update
  this directory)
- On a fresh server / second machine setup

## Full cpbenv rebuild order (from a bare CPlantBox checkout)

If you've lost the entire `cpbenv/`, rebuild in this order:

```bash
# 0. system prereqs (Arch)
sudo pacman -S boost boost-libs cgal eigen qt6-base qt6-5compat cmake

# 1. venv
python3.14 -m venv cpbenv
cpbenv/bin/pip install --upgrade pip setuptools wheel
cpbenv/bin/pip install numpy scipy matplotlib vtk pandas "pybind11[global]" \
    pytest dash plotly opencv-python pillow rasterio

# 2. plantbox C++ extension (in-tree CMake build)
git submodule update --init --recursive   # eigen, pybind11
source cpbenv/bin/activate
cmake .
make -j$(nproc) install                    # installs plantbox.so into cpbenv

# 3. openalea.mtg from GitHub
cpbenv/bin/pip install git+https://github.com/openalea/mtg.git

# 4. openalea.plantgl (this directory)
bash dart/external/plantgl/build.sh

# 5. smoke test
cpbenv/bin/python -c "import plantbox; from openalea.mtg import MTG; \
    from openalea.plantgl.scenegraph import NurbsPatch; print('OK')"
```

Total cost from cold start: ~15-20 min on this machine.
