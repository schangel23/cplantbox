"""
Diurnal meteorological forcing profiles for CPlantBox-DART coupling.

Provides sinusoidal temperature, humidity, and wind profiles for mid-latitude
summer conditions (Jülich, Germany). Optionally loads forcing from CSV.
"""

import csv
import numpy as np
import pandas as pd

from .solar_position import get_solar_positions, sim_day_to_date


def diurnal_met_profile(date, lat=50.92, lon=6.36, freq='30min',
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
