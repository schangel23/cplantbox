# Server Setup Instructions for Claude Code

This file tells Claude Code exactly how to set up the CPlantBox-DART coupling pipeline
on this server from scratch. Read this file fully before taking any action.

## What Is Already Installed

- **Baleno (DART-EB):** Already installed. Find its location with:
  ```bash
  find / -name "dart-eb-main" -type d 2>/dev/null | head -3
  find / -name "dartrc*" -maxdepth 4 2>/dev/null | head -3
  ```
- **DART:** Should be in `~/DART` or nearby. Confirm with `ls ~/DART/bin/`

## What Needs to Be Set Up

1. Python virtual environment (`cpbenv`)
2. CPlantBox compiled `.so` and installed into `cpbenv`
3. pytools4dart installed into `cpbenv`
4. Environment variables configured in `~/.bashrc`

---

## Step 1 — Identify Key Paths

Before doing anything, run these and note the outputs:

```bash
# Where is DART?
ls ~/DART/bin/dart 2>/dev/null || find /home -name "dart" -type f 2>/dev/null | grep bin | head -3

# Where is Baleno (dart-eb-main)?
find / -name "dart-eb-main" -type d 2>/dev/null | head -3

# Where is the dartrc license file?
find ~ -name "dartrc*" -maxdepth 5 2>/dev/null

# Where is Baleno's Python (the venv it uses)?
find / -name "baleno*.py" -type f 2>/dev/null | head -3
find / -path "*/dart-eb-main*" -name "*.py" 2>/dev/null | head -5

# What Python versions are available?
python3 --version
python3.12 --version 2>/dev/null
python3.10 --version 2>/dev/null

# Where is this CPlantBox repo cloned?
pwd
```

Record all paths — you will need them for environment variables.

---

## Step 2 — Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git \
  python3-dev python3-pip python3-venv \
  libvtk9-dev python3-vtk9 \
  pybind11-dev libeigen3-dev \
  libboost-all-dev
```

If `libvtk9-dev` is not available, try `libvtk7-dev` or `libvtk8-dev` depending on Ubuntu version.
Check Ubuntu version with `lsb_release -a`.

---

## Step 3 — Create Python Virtual Environment

Create the venv inside the CPlantBox directory (same structure as local machine):

```bash
cd ~/PHD/CPlantBox   # or wherever CPlantBox is cloned
python3 -m venv cpbenv
source cpbenv/bin/activate
```

Install all Python dependencies:

```bash
pip install --upgrade pip
pip install \
  numpy scipy matplotlib \
  pandas vtk \
  pvlib \
  scikit-learn \
  jupyter ipython
```

---

## Step 4 — Build and Install CPlantBox

With the venv active:

```bash
cd ~/PHD/CPlantBox
mkdir -p build && cd build

cmake .. \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
make install
```

Verify the `.so` was built and installed:

```bash
python3 -c "import plantbox; print('CPlantBox OK')"
```

If that fails, check that `cpbenv/lib/python3.X/site-packages/plantbox/` exists.
If not, manually copy: `cp -r ../src/python_binding/plantbox cpbenv/lib/python3.X/site-packages/`
(replace `python3.X` with the actual version shown by `python3 --version`).

---

## Step 5 — Install pytools4dart

```bash
source ~/PHD/CPlantBox/cpbenv/bin/activate

# Clone if not already present
git clone https://github.com/openalea/pytools4dart.git ~/PHD/pytools4dart

cd ~/PHD/pytools4dart
pip install -e .
```

Verify:
```bash
python3 -c "import pytools4dart as ptd; print('pytools4dart OK')"
```

---

## Step 6 — Configure Environment Variables

Add to `~/.bashrc` (use the actual paths found in Step 1):

```bash
# CPlantBox-DART coupling environment
export DART_HOME=~/DART                           # adjust if DART is elsewhere
export DARTRC=~/.dartrcv1457                      # adjust to actual dartrc filename
export BALENO_PYTHON=<path-to-dart-eb-venv>/bin/python3   # Python used by Baleno

# Python path so `import plantbox` works outside the venv
export PYTHONPATH=~/PHD/CPlantBox/cpbenv/lib/python3.X/site-packages:$PYTHONPATH
```

Then reload: `source ~/.bashrc`

**How to find BALENO_PYTHON:** Baleno has its own Python environment. Look for a
`python3` binary near the `dart-eb-main` directory:
```bash
find / -path "*/dart-eb*" -name "python3" 2>/dev/null
```

---

## Step 7 — Verify the Full Pipeline

```bash
source ~/PHD/CPlantBox/cpbenv/bin/activate
cd ~/PHD/CPlantBox/dart/coupling

# Quick smoke test — imports only, no DART needed
python3 -c "
import plantbox as pb
import pytools4dart as ptd
import numpy as np, scipy, pandas, pvlib
from coupling.geometry.g1_to_g3 import loft_organs
from coupling.geometry.cplantbox_adapter import extract_organs_for_lofter
print('All imports OK')
"
```

---

## Step 8 — Run the Pipeline

The main entry point is `dart/coupling/__main__.py`.

```bash
source ~/PHD/CPlantBox/cpbenv/bin/activate
cd ~/PHD/CPlantBox

# Test geometry only (no DART needed)
python3 -m dart.coupling grow --day 55 --output dart/coupling/output/

# Full DART coupling run (requires DART_HOME and DARTRC set correctly)
python3 -m dart.coupling run --day 55

# Iterative Tuzet-Baleno coupling (Phase 10)
python3 -m dart.coupling iterative --day 55
```

---

## Troubleshooting

### `import plantbox` fails
- Check that `make install` completed without errors
- Try: `find ~/PHD/CPlantBox -name "_plantbox*.so" 2>/dev/null`
- The `.so` must be in the same Python version folder as the venv

### `import vtk` fails
- Install vtk Python bindings: `pip install vtk` (may download pre-built wheel)
- Or ensure system `python3-vtk9` is accessible from the venv

### DART simulation fails / DARTRC error
- DART requires a valid license file. Confirm `$DARTRC` points to the correct file
- Test: `$DART_HOME/bin/dart -version`

### pvlib not found
- `pip install pvlib` inside the venv

### Wrong Python version in `.so`
- CPlantBox `.so` is compiled for a specific Python version
- The venv Python must match: `python3 --version` inside venv must match build Python
- Rebuild with `cmake -DPYTHON_EXECUTABLE=$(which python3)` pointing to venv Python

---

## Directory Layout After Setup

```
~/PHD/
├── CPlantBox/
│   ├── cpbenv/             ← Python venv (create here)
│   ├── build/              ← CMake build dir
│   ├── dart/coupling/      ← Pipeline entry point
│   │   ├── config.py       ← Edit DART_HOME/BALENO_PYTHON if env vars don't work
│   │   └── output/         ← Results go here
│   └── src/                ← CPlantBox C++ source (already patched)
└── pytools4dart/           ← Clone here
```

## Config Override (if env vars are awkward)

If setting env vars is difficult, edit `dart/coupling/config.py` directly:

```python
DART_HOME = Path("/actual/path/to/DART")
DART_EB_DIR = Path("/actual/path/to/dart-eb-main")
DARTRC = Path("/actual/path/to/.dartrcv1457")
BALENO_PYTHON = Path("/actual/path/to/venv/bin/python3")
```
