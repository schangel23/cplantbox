"""
Solar position calculator for diurnal coupling loop.

Uses pvlib for accurate sun zenith/azimuth. Default location comes from
the active site config (COUPLING_SITE env var, default Jülich).
"""

import pandas as pd
import pvlib

from ..config import DEFAULT_LAT, DEFAULT_LON, DEFAULT_ALTITUDE, DEFAULT_SOWING_DATE


def get_solar_positions(date, lat=DEFAULT_LAT, lon=DEFAULT_LON,
                        altitude=DEFAULT_ALTITUDE, freq='30min'):
    """Return DataFrame with zenith, azimuth for each timestep on given date.

    Only returns daylight hours (apparent_zenith < 90°).

    Args:
        date: datetime.date or string 'YYYY-MM-DD'.
        lat: Latitude in degrees North.
        lon: Longitude in degrees East.
        altitude: Elevation in meters.
        freq: Timestep frequency (pandas offset alias, e.g. '30min', '60min').

    Returns:
        pd.DataFrame with columns: apparent_zenith, azimuth, apparent_elevation.
        Index is timezone-aware DatetimeIndex (UTC).
    """
    location = pvlib.location.Location(lat, lon, tz='UTC', altitude=altitude)

    # Create time range for the full day
    times = pd.date_range(
        start=f'{date} 00:00', end=f'{date} 23:59',
        freq=freq, tz='UTC',
    )

    solpos = location.get_solarposition(times)

    # Filter to daylight (sun above horizon)
    daylight = solpos[solpos['apparent_zenith'] < 90].copy()

    return daylight[['apparent_zenith', 'azimuth', 'apparent_elevation']]


def sim_day_to_date(sim_day, sowing_date=DEFAULT_SOWING_DATE):
    """Convert CPlantBox simulation day to calendar date.

    Args:
        sim_day: Days since sowing (CPlantBox simulation time).
        sowing_date: Sowing date string 'YYYY-MM-DD'.

    Returns:
        datetime.date object.
    """
    sowing = pd.Timestamp(sowing_date)
    return (sowing + pd.Timedelta(days=int(sim_day))).date()


def get_clearsky_par(time, lat=DEFAULT_LAT, lon=DEFAULT_LON,
                     altitude=DEFAULT_ALTITUDE):
    """Compute clear-sky PAR irradiance at a specific time.

    Uses pvlib's Ineichen clear-sky model to get GHI, then converts to PAR.

    Args:
        time: pandas Timestamp (timezone-aware, UTC).
        lat: Latitude in degrees North.
        lon: Longitude in degrees East.
        altitude: Elevation in meters.

    Returns:
        Clear-sky PAR irradiance in W/m² (on horizontal surface).
    """
    location = pvlib.location.Location(lat, lon, tz='UTC', altitude=altitude)

    if not hasattr(time, '__iter__'):
        times = pd.DatetimeIndex([time])
    else:
        times = pd.DatetimeIndex(time)

    cs = location.get_clearsky(times, model='ineichen')
    ghi = float(cs['ghi'].iloc[0])

    # PAR is approximately 45% of GHI (McCree 1972, Meek et al. 1984)
    par_wm2 = max(ghi * 0.45, 0.0)
    return par_wm2


def day_of_year(sim_day, sowing_date=DEFAULT_SOWING_DATE):
    """Convert CPlantBox simulation day to day-of-year (1-365).

    Args:
        sim_day: Days since sowing.
        sowing_date: Sowing date string 'YYYY-MM-DD'.

    Returns:
        int day of year.
    """
    d = sim_day_to_date(sim_day, sowing_date)
    return d.timetuple().tm_yday
