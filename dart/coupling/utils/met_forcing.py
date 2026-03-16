"""
Diurnal meteorological forcing profiles for CPlantBox-DART coupling.

Provides sinusoidal temperature, humidity, and wind profiles.
Optionally loads forcing from CSV or FLUXNET hourly files.
Default location comes from the active site config.
"""

import csv
import numpy as np
import pandas as pd

from .solar_position import get_solar_positions, sim_day_to_date
from ..config import DEFAULT_LAT, DEFAULT_LON, DEFAULT_SOWING_DATE


def diurnal_met_profile(date, lat=DEFAULT_LAT, lon=DEFAULT_LON, freq='30min',
                        T_min=18.0, T_max=30.0, RH_min=0.40, RH_max=0.80,
                        wind_min=1.0, wind_max=3.0):
    """Generate diurnal meteorological forcing for a given date.

    Temperature follows a sinusoidal curve peaking at 14:00 local solar time
    (~12:30 UTC at Jülich). RH is inversely correlated with temperature.
    Wind has a gentle afternoon peak.

    Args:
        date: datetime.date or 'YYYY-MM-DD' string.
        lat, lon: Location coordinates.
        freq: Time resolution (pandas offset alias).
        T_min, T_max: Temperature range (°C).
        RH_min, RH_max: Relative humidity range (0-1).
        wind_min, wind_max: Wind speed range (m/s).

    Returns:
        pd.DataFrame indexed by UTC time with columns:
            T_air_C, RH, wind_ms, ea_hPa, es_hPa, T_air_K
    """
    # Get solar positions to determine daylight hours
    solar = get_solar_positions(date, lat, lon, freq=freq)
    if solar.empty:
        return pd.DataFrame()

    times = solar.index

    # Compute fractional hour of day (UTC)
    hours = np.array([t.hour + t.minute / 60.0 for t in times])

    # Temperature: sinusoidal, peak at ~12.5 UTC (≈14:00 local at Jülich)
    # T(h) = T_mean + T_amp * sin(2π(h - 6.5)/24) where peak at h=12.5
    T_mean = (T_min + T_max) / 2.0
    T_amp = (T_max - T_min) / 2.0
    T_air = T_mean + T_amp * np.sin(2 * np.pi * (hours - 6.5) / 24.0)
    T_air = np.clip(T_air, T_min, T_max)

    # RH: inverse correlation with temperature
    # High in morning (T low), low in afternoon (T high)
    T_norm = (T_air - T_min) / max(T_max - T_min, 1e-6)
    RH = RH_max - (RH_max - RH_min) * T_norm

    # Wind: gentle afternoon peak (sinusoidal, peak at 14 UTC)
    wind = wind_min + (wind_max - wind_min) * (
        0.5 + 0.5 * np.sin(2 * np.pi * (hours - 8.0) / 24.0)
    )
    wind = np.clip(wind, wind_min, wind_max)

    # Derived quantities
    # Saturation vapour pressure (Tetens formula)
    es_hPa = 6.1078 * np.exp(17.269 * T_air / (T_air + 237.3))
    ea_hPa = RH * es_hPa
    T_air_K = T_air + 273.15

    met = pd.DataFrame({
        'T_air_C': T_air,
        'T_air_K': T_air_K,
        'RH': RH,
        'wind_ms': wind,
        'es_hPa': es_hPa,
        'ea_hPa': ea_hPa,
    }, index=times)

    return met


def load_met_csv(filepath):
    """Load meteorological forcing from a CSV file.

    Expected columns: datetime_utc, T_air_C, RH, wind_ms
    Optional columns: ea_hPa, PAR_umol

    Args:
        filepath: Path to CSV file.

    Returns:
        pd.DataFrame with UTC DatetimeIndex and met columns.
    """
    df = pd.read_csv(filepath, parse_dates=['datetime_utc'])
    df = df.set_index('datetime_utc')
    df.index = df.index.tz_localize('UTC')

    # Compute derived quantities if missing
    if 'es_hPa' not in df.columns:
        df['es_hPa'] = 6.1078 * np.exp(
            17.269 * df['T_air_C'] / (df['T_air_C'] + 237.3)
        )
    if 'ea_hPa' not in df.columns:
        df['ea_hPa'] = df['RH'] * df['es_hPa']
    if 'T_air_K' not in df.columns:
        df['T_air_K'] = df['T_air_C'] + 273.15

    return df


def derive_daily_met_from_csv(filepath, sowing_date=DEFAULT_SOWING_DATE):
    """Derive daily T_min/T_max/T_mean from an hourly met CSV.

    Aggregates by calendar date, maps to sim_day, and injects into
    the DVS daily_met cache so GDD uses the same weather source.

    Args:
        filepath: Path to hourly met CSV (datetime_utc, T_air_C, RH, wind_ms).
        sowing_date: Sowing date string for sim_day mapping.

    Returns:
        dict mapping sim_day -> {T_mean_C, T_min_C, T_max_C, ...}
    """
    import datetime

    df = pd.read_csv(filepath, parse_dates=['datetime_utc'])
    df['date'] = df['datetime_utc'].dt.date

    sowing = datetime.date.fromisoformat(sowing_date)

    daily = df.groupby('date').agg(
        T_min_C=('T_air_C', 'min'),
        T_max_C=('T_air_C', 'max'),
        T_mean_C=('T_air_C', 'mean'),
    )

    # Add RH and wind if available
    if 'RH' in df.columns:
        rh_stats = df.groupby('date')['RH'].agg(['min', 'max'])
        daily['RH_min'] = rh_stats['min']
        daily['RH_max'] = rh_stats['max']
    if 'wind_ms' in df.columns:
        wind_stats = df.groupby('date')['wind_ms'].agg(['mean', 'max'])
        daily['wind_mean_ms'] = wind_stats['mean']
        daily['wind_max_ms'] = wind_stats['max']

    result = {}
    for cal_date, row in daily.iterrows():
        if isinstance(cal_date, str):
            cal_date = datetime.date.fromisoformat(cal_date)
        sim_day = (cal_date - sowing).days + 1
        if sim_day < 1:
            continue
        entry = {
            'T_mean_C': row['T_mean_C'],
            'T_min_C': row['T_min_C'],
            'T_max_C': row['T_max_C'],
        }
        if 'RH_min' in row:
            entry['RH_min'] = row['RH_min']
            entry['RH_max'] = row['RH_max']
        if 'wind_mean_ms' in row:
            entry['wind_mean_ms'] = row['wind_mean_ms']
            entry['wind_max_ms'] = row['wind_max_ms']
        result[sim_day] = entry

    return result


def inject_met_csv_into_dvs(filepath, sowing_date=DEFAULT_SOWING_DATE):
    """Load hourly met CSV, derive daily stats, merge into DVS cache.

    Days present in the CSV override the default Jülich data.
    Days not in the CSV fall back to the default.
    """
    from ..carbon.dvs_partitioning import get_daily_met, _DEFAULT_DAILY_MET
    import dart.coupling.carbon.dvs_partitioning as dvs_mod

    csv_daily = derive_daily_met_from_csv(filepath, sowing_date)

    # Load existing defaults
    existing = get_daily_met() or {}

    # Merge: CSV overrides defaults
    merged = dict(existing)
    merged.update(csv_daily)

    # Inject into global cache
    dvs_mod._DEFAULT_DAILY_MET = merged

    n_override = len(csv_daily)
    print(f"  Met CSV: injected {n_override} days into DVS daily met cache")


# ---------------------------------------------------------------------------
# FLUXNET hourly data loader
# ---------------------------------------------------------------------------

def load_fluxnet_csv(filepath, year):
    """Load FLUXNET FULLSET hourly CSV and convert to pipeline format.

    Reads the AmeriFlux/FLUXNET FULLSET_HR file, filters to a single year,
    and converts column names and units to the pipeline's internal format.

    Args:
        filepath: Path to FLUXNET HR CSV file.
        year: Year to extract (int).

    Returns:
        pd.DataFrame with UTC DatetimeIndex and columns:
            T_air_C, RH, wind_ms, ea_hPa, es_hPa, T_air_K,
            PAR_umol, SW_in_Wm2, LW_in_Wm2, P_hPa, CO2_ppm
    """
    # Read only needed columns for performance (~210k rows total)
    usecols = [
        'TIMESTAMP_START', 'TA_F', 'SW_IN_F', 'VPD_F', 'PA_F',
        'WS_F', 'RH', 'P_F', 'LW_IN_F', 'CO2_F_MDS',
    ]
    # Also grab PPFD_IN if available (measured PAR)
    # Read header first to check
    with open(filepath) as f:
        header = f.readline().strip().split(',')
    if 'PPFD_IN' in header:
        usecols.append('PPFD_IN')

    df = pd.read_csv(filepath, usecols=usecols)

    # Parse timestamp and filter to year
    df['datetime_utc'] = pd.to_datetime(df['TIMESTAMP_START'], format='%Y%m%d%H%M')
    df = df[df['datetime_utc'].dt.year == year].copy()
    df = df.set_index('datetime_utc')
    df.index = df.index.tz_localize('UTC')

    if df.empty:
        raise ValueError(f"No data found for year {year} in {filepath}")

    # Replace FLUXNET missing value (-9999) with NaN
    df = df.replace(-9999, np.nan).replace(-9999.0, np.nan)

    # Forward-fill short gaps (up to 3 hours)
    df = df.ffill(limit=3)

    # --- Unit conversions ---
    out = pd.DataFrame(index=df.index)

    # Temperature: already °C
    out['T_air_C'] = df['TA_F']
    out['T_air_K'] = df['TA_F'] + 273.15

    # RH: FLUXNET gives 0-100%, pipeline expects 0-1 fraction
    out['RH'] = df['RH'] / 100.0

    # Wind speed: already m/s
    out['wind_ms'] = df['WS_F']

    # Vapour pressure: es from Tetens, ea = es - VPD
    out['es_hPa'] = 6.1078 * np.exp(17.269 * df['TA_F'] / (df['TA_F'] + 237.3))
    out['ea_hPa'] = out['es_hPa'] - df['VPD_F']  # VPD_F already in hPa

    # PAR: prefer measured PPFD_IN if available, else derive from SW_IN
    if 'PPFD_IN' in df.columns and df['PPFD_IN'].notna().sum() > 100:
        out['PAR_umol'] = df['PPFD_IN']
    else:
        # SW_IN (W/m²) -> PAR (µmol/m²/s): PAR ≈ 48% of SW, 4.57 µmol/J
        out['PAR_umol'] = df['SW_IN_F'] * 0.48 * 4.57

    # Additional columns (pass-through)
    out['SW_in_Wm2'] = df['SW_IN_F']
    out['LW_in_Wm2'] = df['LW_IN_F']
    out['P_hPa'] = df['PA_F'] * 10.0  # kPa -> hPa
    out['CO2_ppm'] = df['CO2_F_MDS']

    # Clamp negative PAR to 0 (nighttime rounding)
    out['PAR_umol'] = out['PAR_umol'].clip(lower=0.0)

    n_valid = out['T_air_C'].notna().sum()
    n_total = len(out)
    print(f"  FLUXNET: loaded {n_valid}/{n_total} hourly rows for {year}")

    return out


def load_fluxnet_validation(filepath, year):
    """Load FLUXNET flux columns for validation comparison.

    Returns GPP, NEE, Reco, LE, H at hourly resolution for the given year.

    Args:
        filepath: Path to FLUXNET HR CSV file.
        year: Year to extract (int).

    Returns:
        pd.DataFrame with UTC DatetimeIndex and columns:
            GPP_gC_m2_d, NEE_gC_m2_d, Reco_gC_m2_d, LE_Wm2, H_Wm2
    """
    usecols = [
        'TIMESTAMP_START', 'GPP_NT_VUT_REF', 'NEE_VUT_REF',
        'RECO_NT_VUT_REF', 'LE_F_MDS', 'H_F_MDS',
    ]
    df = pd.read_csv(filepath, usecols=usecols)
    df['datetime_utc'] = pd.to_datetime(df['TIMESTAMP_START'], format='%Y%m%d%H%M')
    df = df[df['datetime_utc'].dt.year == year].copy()
    df = df.set_index('datetime_utc')
    df.index = df.index.tz_localize('UTC')
    df = df.replace(-9999, np.nan).replace(-9999.0, np.nan)

    out = pd.DataFrame(index=df.index)
    # FLUXNET GPP/NEE/Reco are in gC m-2 d-1 (daily rate at each timestep)
    out['GPP_gC_m2_d'] = df['GPP_NT_VUT_REF']
    out['NEE_gC_m2_d'] = df['NEE_VUT_REF']
    out['Reco_gC_m2_d'] = df['RECO_NT_VUT_REF']
    out['LE_Wm2'] = df['LE_F_MDS']
    out['H_Wm2'] = df['H_F_MDS']

    return out


def fluxnet_to_pipeline_csv(fluxnet_path, year, output_path):
    """Convert FLUXNET hourly CSV to pipeline-format met CSV for one year.

    Writes a CSV with columns: datetime_utc, T_air_C, RH, wind_ms, ea_hPa,
    PAR_umol — compatible with load_met_csv().

    Args:
        fluxnet_path: Path to FLUXNET HR CSV.
        year: Year to extract.
        output_path: Where to write the pipeline-format CSV.

    Returns:
        Path to the output CSV (same as output_path).
    """
    df = load_fluxnet_csv(fluxnet_path, year)
    out = df[['T_air_C', 'RH', 'wind_ms', 'ea_hPa', 'PAR_umol']].copy()
    out.index.name = 'datetime_utc'
    # Remove timezone info for CSV (load_met_csv re-localizes)
    out.index = out.index.tz_localize(None)
    out.to_csv(output_path)
    print(f"  Wrote pipeline met CSV: {output_path} ({len(out)} rows)")
    return output_path


def inject_fluxnet_into_dvs(fluxnet_path, year, sowing_date=DEFAULT_SOWING_DATE):
    """Load FLUXNET hourly, derive daily T stats, inject into DVS cache."""
    import datetime
    from ..carbon.dvs_partitioning import get_daily_met
    import dart.coupling.carbon.dvs_partitioning as dvs_mod

    df = load_fluxnet_csv(fluxnet_path, year)
    df['date'] = df.index.date

    sowing = datetime.date.fromisoformat(sowing_date)

    daily = df.groupby('date').agg(
        T_min_C=('T_air_C', 'min'),
        T_max_C=('T_air_C', 'max'),
        T_mean_C=('T_air_C', 'mean'),
    )
    rh_stats = df.groupby('date')['RH'].agg(['min', 'max'])
    daily['RH_min'] = rh_stats['min']
    daily['RH_max'] = rh_stats['max']
    wind_stats = df.groupby('date')['wind_ms'].agg(['mean', 'max'])
    daily['wind_mean_ms'] = wind_stats['mean']
    daily['wind_max_ms'] = wind_stats['max']

    result = {}
    for cal_date, row in daily.iterrows():
        sim_day = (cal_date - sowing).days + 1
        if sim_day < 1:
            continue
        result[sim_day] = {
            'T_mean_C': row['T_mean_C'],
            'T_min_C': row['T_min_C'],
            'T_max_C': row['T_max_C'],
            'RH_min': row['RH_min'],
            'RH_max': row['RH_max'],
            'wind_mean_ms': row['wind_mean_ms'],
            'wind_max_ms': row['wind_max_ms'],
        }

    existing = get_daily_met() or {}
    merged = dict(existing)
    merged.update(result)
    dvs_mod._DEFAULT_DAILY_MET = merged

    print(f"  FLUXNET: injected {len(result)} days into DVS daily met cache")
