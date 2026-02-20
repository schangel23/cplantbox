# CPlantBox → DART Coupling Guide

Complete workflow for coupling CPlantBox photosynthesis with DART radiative transfer, using Baker et al. (2025) validated approach.

## Quick Start

```python
import plantbox as pb
from plantbox.visualisation.dart_coupling import (
    export_plant_for_dart,
    map_dart_to_cplantbox_segments
)

# 1. Create and simulate plant
plant = pb.MappedPlant()
plant.readParameters("path/to/params.xml")
plant.initialize()
plant.simulate(30)

# 2. Export for DART
vis = pb.PlantVisualiser(plant)
export_plant_for_dart(vis, plant, "plant.obj")

# 3. Run DART simulation (see DART workflow below)

# 4. Map DART results back to CPlantBox
dart_apar_wm2 = load_dart_triangle_results("radiativeBudget_Triangles.nc")  # W/m²
segment_ppfd = map_dart_to_cplantbox_segments(
    dart_apar_wm2, "plant_mapping.json", plant, convert_units=True
)  # Automatically converts W/m² → mol photons cm⁻² d⁻¹

# 5. Feed to photosynthesis
from plantbox.functional.Photosynthesis import PhotosynthesisPython
photo = PhotosynthesisPython(plant, params)
photo.Qlight = segment_ppfd  # Already in correct units!
photo.solve(...)
An = photo.get_net_assimilation()
```

## Files Created

### 1. `plant.obj`
Wavefront OBJ file with per-organ groups:
```obj
g organ_5    # Stem
g organ_7    # Leaf 1
g organ_12   # Leaf 2
```

### 2. `plant_mapping.json`
Triangle-to-segment mapping:
```json
{
  "total_triangles": 15234,
  "total_segments": 450,
  "triangle_to_segment": {
    "0": {
      "organ_id": 5,
      "local_segment_index": 0,
      "global_segment_index": 0,
      "node_ids": [12, 13, 14],
      "segment_nodes": [12, 13]
    },
    ...
  }
}
```

## DART Workflow

### Using pytools4dart:

```python
import pytools4dart as ptd

# 1. Create DART simulation
simu = ptd.simulation.Simulation()

# 2. Import plant geometry
simu.add.object_3d(
    "plant.obj",
    position=(0, 0, 0),
    scale=0.01  # cm to m
)

# 3. Assign optical properties (PROSPECT model)
simu.add.optical_property(
    type="leaf",
    ident="leaf_prospect",
    prospect={
        'CBrown': 0.0,
        'Cab': 40.0,      # Chlorophyll a+b [μg cm⁻²]
        'Car': 8.0,       # Carotenoids [μg cm⁻²]
        'Cm': 0.009,      # Dry matter [g cm⁻²]
        'Cw': 0.012,      # Water [cm]
        'N': 1.5          # Structure parameter
    }
)

# 4. Set illumination
simu.core.phase.Phase.ExpertModeZone.dart.sunViewingAngles.sunViewingAngles = {
    'sunZenithAngle': 30,   # degrees
    'sunAzimuthAngle': 135
}

# 5. Run simulation
simu.run()

# 6. Extract per-triangle APAR
import netCDF4 as nc
dart_output = nc.Dataset(simu.output_dir / "radiativeBudget_Triangles.nc")
triangle_apar = dart_output.variables['aPAR'][:]  # μmol m⁻² s⁻¹
```

## Understanding the Mapping

### Export creates per-organ groups:
```
Organ 5 (stem) → 200 triangles → 3 segments
  Triangle 0-66   → Segment 0 (nodes 12→13)
  Triangle 67-133 → Segment 1 (nodes 13→14)
  Triangle 134-199 → Segment 2 (nodes 14→15)
```

### DART calculates per-triangle APAR:
```
Triangle 0: 450.2 μmol m⁻² s⁻¹
Triangle 1: 448.9
Triangle 2: 451.3
...
```

### Mapping aggregates to segments:
```python
# Triangles 0-66 all map to segment 0
segment_0_apar = mean([450.2, 448.9, 451.3, ...])  # Average of 67 triangles
```

### Result matches CPlantBox segment array:
```python
segment_apar = [450.1, 380.5, 420.3, ...]  # One value per segment
                 ^seg0   ^seg1   ^seg2
```

## Why Per-Organ Grouping (Baker's Approach)

1. **Uses existing CPlantBox APIs** - `WavefrontFromPlantGeometry()` already does per-organ grouping
2. **Proven by Baker et al.** - validated against field data
3. **Correct segment order** - uses MappedPlant's internal global ordering
4. **Flexible** - DART calculates per-triangle, aggregate to segments via node IDs

## Unit Conversions

### DART → CPlantBox Photosynthesis:

**CRITICAL:** DART outputs APAR in **W/m²** (radiometric irradiance), NOT μmol m⁻² s⁻¹!

```python
# DART outputs: W/m² (radiometric energy per triangle)
# CPlantBox needs: mol photons cm⁻² d⁻¹ (per segment)

# Conversion (done automatically by dart_coupling.py):
from plantbox.visualisation.dart_coupling import convert_dart_to_cplantbox_units

ppfd_daily = convert_dart_to_cplantbox_units(dart_apar_wm2)

# Manual breakdown:
# 1. W/m² → μmol photons m⁻² s⁻¹:  multiply by 4.6 (PAR conversion factor)
# 2. μmol m⁻² s⁻¹ → mol m⁻² s⁻¹:   divide by 1e6
# 3. mol m⁻² s⁻¹ → mol m⁻² d⁻¹:    multiply by 86400 (seconds per day)
# 4. mol m⁻² d⁻¹ → mol cm⁻² d⁻¹:   divide by 10000 (cm² per m²)

# Combined:
ppfd_daily = dart_apar_wm2 * 4.6 / 1e6 * 86400 / 10000
```

**Notes:**
- Factor 4.6 μmol photons/Joule is typical for PAR (400-700 nm) under sunlight
- See McCree (1972), Thimijan & Heins (1983) for PAR conversion factors
- DART's spectral output allows more precise wavelength-specific conversion if needed

## Troubleshooting

### "Segment count mismatch"
- Plant structure changed between export and feedback
- Re-export the plant: `export_plant_for_dart(vis, plant, "plant.obj")`

### "Triangle not in DART results"
- DART output has fewer triangles than exported
- Check DART import was successful
- Verify OBJ file integrity

### "Order doesn't match photosynthesis"
- Using regular `Plant` instead of `MappedPlant`
- Solution: Use `MappedPlant` for photosynthesis coupling

## References

Baker et al. (2025). Bi-directional coupling between CPlantBox functional-structural
plant model and Unreal Engine for digital twin of agricultural environments.
*Validated approach for photosynthesis-radiative transfer coupling.*

## Files in This Package

- `dart_coupling.py` - Main coupling functions (Baker's approach)
- `test_dart_coupling.py` - Test suite and validation
- `DART_COUPLING_README.md` - This file
- `WORKFLOW_SUMMARY.md` - Complete workflow overview

## See Also

- CPlantBox documentation: https://github.com/Plant-Root-Soil-Interactions-Modelling/CPlantBox
- pytools4dart: https://pytools4dart.readthedocs.io/
- DART model: https://dart.omp.eu/
