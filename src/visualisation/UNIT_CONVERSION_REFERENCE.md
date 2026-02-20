# DART ↔ CPlantBox Unit Conversion Reference

## Quick Reference

```
DART Output:         W/m² (radiometric irradiance)
                        ↓
Conversion Factor:   × 4.6 μmol photons/Joule
                        ↓
PPFD:               μmol photons m⁻² s⁻¹
                        ↓
Time Integration:    × 86400 s/day
                        ↓
Daily PPFD:         μmol photons m⁻² d⁻¹
                        ↓
Unit Conversion:     ÷ 1e6 (μmol → mol) ÷ 10000 (m² → cm²)
                        ↓
CPlantBox Input:    mol photons cm⁻² d⁻¹
```

## Detailed Conversion

### DART → CPlantBox (Automatic)

```python
from plantbox.visualisation.dart_coupling import convert_dart_to_cplantbox_units

# DART outputs APAR in W/m²
dart_apar_wm2 = 150.0  # W/m²

# Automatic conversion
ppfd_mol_cm2_d = convert_dart_to_cplantbox_units(dart_apar_wm2)
# Result: 0.0597 mol photons cm⁻² d⁻¹
```

### Step-by-Step Manual Conversion

```python
import numpy as np

# Starting value from DART
apar_wm2 = 150.0  # W/m²

# Step 1: W/m² → μmol photons m⁻² s⁻¹
# PAR conversion factor: ~4.6 μmol photons per Joule for sunlight (400-700 nm)
PAR_FACTOR = 4.6  # μmol photons/J
ppfd_umol_m2_s = apar_wm2 * PAR_FACTOR  # = 690.0 μmol m⁻² s⁻¹

# Step 2: μmol → mol
ppfd_mol_m2_s = ppfd_umol_m2_s / 1e6  # = 0.00069 mol m⁻² s⁻¹

# Step 3: per second → per day
ppfd_mol_m2_d = ppfd_mol_m2_s * 86400  # = 59.616 mol m⁻² d⁻¹

# Step 4: m² → cm²
ppfd_mol_cm2_d = ppfd_mol_m2_d / 10000  # = 0.0597 mol cm⁻² d⁻¹
```

### One-Line Formula

```python
ppfd_mol_cm2_d = apar_wm2 * 4.6 / 1e6 * 86400 / 10000
# Simplified:
ppfd_mol_cm2_d = apar_wm2 * 0.00039744
```

## Typical Value Ranges

| Source | APAR (W/m²) | PPFD (μmol m⁻² s⁻¹) | PPFD (mol cm⁻² d⁻¹) |
|--------|-------------|----------------------|----------------------|
| Full sun | 200-400 | 920-1840 | 0.080-0.159 |
| Moderate shade | 100-200 | 460-920 | 0.040-0.080 |
| Deep shade | 20-100 | 92-460 | 0.008-0.040 |
| DART typical | 50-300 | 230-1380 | 0.020-0.119 |

## PAR Conversion Factor Details

The factor **4.6 μmol photons/Joule** is commonly used for PAR (400-700 nm) because:

1. **Spectral distribution matters**: Different wavelengths carry different energy per photon
2. **Sunlight assumption**: Factor assumes typical solar spectrum in PAR range
3. **Literature values**:
   - McCree (1972): 4.57 μmol/J for sunlight
   - Thimijan & Heins (1983): 4.6 μmol/J (standard)
   - Sager et al. (1982): 4.66 μmol/J

### More Precise Conversion (Advanced)

If DART outputs spectral data, you can compute wavelength-specific conversion:

```python
def photon_energy(wavelength_nm):
    """Energy per photon in Joules."""
    h = 6.62607015e-34  # Planck constant (J·s)
    c = 299792458       # Speed of light (m/s)
    return h * c / (wavelength_nm * 1e-9)

# For 550 nm (middle of PAR):
E_550 = photon_energy(550)  # = 3.61e-19 J/photon
# 1 J = 1 / 3.61e-19 photons = 2.77e18 photons = 4.60 μmol photons
```

## Validation Examples

### Example 1: Clear sky at noon
```python
dart_apar = 250.0  # W/m² (sunny day, well-lit leaf)
ppfd = convert_dart_to_cplantbox_units(dart_apar)
# Result: 0.0994 mol photons cm⁻² d⁻¹
# ✓ Reasonable for full sun
```

### Example 2: Shaded leaf
```python
dart_apar = 80.0  # W/m² (shaded by upper canopy)
ppfd = convert_dart_to_cplantbox_units(dart_apar)
# Result: 0.0318 mol photons cm⁻² d⁻¹
# ✓ Reasonable for moderate shade
```

### Example 3: Deep canopy
```python
dart_apar = 30.0  # W/m² (deep inside canopy)
ppfd = convert_dart_to_cplantbox_units(dart_apar)
# Result: 0.0119 mol photons cm⁻² d⁻¹
# ✓ Reasonable for deep shade
```

## Units Summary Table

| Quantity | Symbol | Units | Where Used |
|----------|--------|-------|------------|
| Irradiance (radiometric) | E | W/m² | DART output |
| Photon flux (instantaneous) | PPFD | μmol photons m⁻² s⁻¹ | Intermediate |
| Daily photon flux (area) | Qₗᵢₘₕₜ | mol photons m⁻² d⁻¹ | Intermediate |
| Daily photon flux (CPlantBox) | Qₗᵢₘₕₜ | mol photons cm⁻² d⁻¹ | Photosynthesis input |
| Net assimilation | Aₙ | mol CO₂ m⁻² s⁻¹ | Standard units |
| Net assimilation (CPlantBox) | Aₙ | mol CO₂ d⁻¹ | Per segment output |

## Code Usage

### In dart_coupling.py:

```python
# Automatic conversion during mapping
segment_ppfd = map_dart_to_cplantbox_segments(
    dart_triangle_apar_wm2,           # W/m² from DART
    "plant_mapping.json",
    plant,
    convert_units=True                 # Automatically converts
)
# segment_ppfd is now in mol photons cm⁻² d⁻¹
```

### Direct conversion:

```python
from plantbox.visualisation.dart_coupling import convert_dart_to_cplantbox_units

# If you have APAR in W/m² and need conversion
dart_wm2 = np.array([150.0, 200.0, 80.0])
ppfd = convert_dart_to_cplantbox_units(dart_wm2)
# ppfd = [0.0597, 0.0795, 0.0318] mol photons cm⁻² d⁻¹
```

## References

- McCree, K. J. (1972). Test of current definitions of photosynthetically active radiation against leaf photosynthesis data. *Agricultural Meteorology*, 10, 443-453.
- Thimijan, R. W., & Heins, R. D. (1983). Photometric, radiometric, and quantum light units of measure: a review of procedures for interconversion. *HortScience*, 18(6), 818-822.
- Sager, J. C., Smith, W. O., Edwards, J. L., & Cyr, K. L. (1982). Photosynthetic efficiency and phytochrome photoequilibria determination using spectral data. *Transactions of the ASAE*, 25(6), 1882-1889.
