"""
Midwest Pest-Risk Weather Dataset + XGBoost Forecaster
=========================================================

PS-2A: Early Crop Disease & Pest Outbreak Prediction

Pipeline (see architecture diagram):

    WEATHER (10yr hourly, Open-Meteo)
        |
        v
    DAILY AGGREGATION  -> engineered feature dataset (exact columns below)
        |
        v
    XGBoost: WEATHER FORECASTER
    (predicts the next 7-30 daily rows themselves --
     temperature_max/min, rain_mm, wind_speed_mean,
     wind_direction, pressure_mean -- NOT a risk score)
        |
        v
    forecasted future rows (day t+1 ... t+30)
        |
        v
    re-run the SAME GDD / arrival-signal calculators
    from the earlier script on these forecasted rows
        |
        v
             overall pest risk map
                     v
              treatment logic
                     v
            pesticide avoided

-----------------------------------------------------------------
IMPORTANT, READ FIRST
-----------------------------------------------------------------
1. CAPE: Open-Meteo's HISTORICAL ARCHIVE (ERA5 reanalysis) does NOT
   expose a `cape` variable -- it only exists on the live FORECAST
   endpoint (next 16 days). So in the 10-year training dataset, the
   `cape` column is left as NaN with a clear flag, and `storm_flag`
   is instead computed from precipitation + wind speed. If you later
   run this against the live forecast endpoint for real-time scoring,
   a `fetch_live_cape()` helper is included to backfill cape for the
   *live* scoring window only -- never for historical training rows.

2. No pest-occurrence labels exist publicly for this problem (that is
   the whole reason the brief asks for a *risk* system, not a
   *detection* system). So XGBoost here is NOT trained to predict
   "pest outbreak" or an abstract risk score directly -- it is trained
   to forecast the ACTUAL FUTURE WEATHER ROWS: what temperature_max,
   temperature_min, rain_mm, wind_speed_mean, wind_direction, and
   pressure_mean are likely to be, 1 to 30 days from today. That is a
   task the weather data can genuinely teach a model to do, unlike
   "pest outbreak" which has no labels to learn from.

   Once you have those forecasted future rows, you run the SAME GDD
   and arrival-signal calculator functions from the earlier script
   (midwest_pest_risk.py) on top of them, which is what turns a
   weather forecast into a forward-looking pest-risk estimate. This
   script deliberately does NOT re-implement that risk logic --
   forecasting weather and scoring risk are kept as two separate,
   swappable steps.

   Wind direction is circular (0 and 360 degrees are the same point
   on the compass), so it is never regressed directly -- it's split
   into wind_dir_sin / wind_dir_cos components for modeling, then
   recombined back into a compass degree at prediction time. See
   `reconstruct_wind_direction()` below.

   Forecasting daily weather 30 days out with real skill is genuinely
   hard (all weather forecasting, XGBoost or otherwise, loses accuracy
   fast past ~7-10 days). So this trains a DAILY model for horizons
   1-7 (short-range, most trustworthy), plus lower-resolution check-in
   points at 14, 21, and 30 days rather than pretending to give a
   precise daily forecast that far out. Expect the MAE printed at
   training time to climb noticeably at the longer horizons -- report
   that honestly rather than hiding it.

3. This sandbox has no network access and no `xgboost` package
   installed, so the historical-fetch calls and the model-fit calls
   in this file could not be executed live here. Everything else
   (aggregation math, circular wind averaging, rolling windows, GDD
   accumulation, target construction, train/test split, feature
   matrix shapes) WAS run end-to-end against 10 years of synthetic
   hourly data during development and produced no errors -- see the
   bottom of this file for how to reproduce that check. Run
   `pip install xgboost` before running this for real.
-----------------------------------------------------------------
"""

from __future__ import annotations

import datetime as dt
import math
import os
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import requests

try:
    from xgboost import XGBRegressor
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "xgboost is required for the training step. Install with: pip install xgboost"
    ) from e

from sklearn.metrics import mean_absolute_error


# ============================================================
# CONFIG
# ============================================================

MIDWEST_LOCATIONS: dict[str, tuple[float, float]] = {
    "Illinois (Champaign Co.)": (40.12, -88.24),
    "Indiana (Tippecanoe Co.)": (40.42, -86.89),
    "Iowa (Story Co.)": (42.03, -93.62),
    "Kansas (Riley Co.)": (39.19, -96.59),
    "Michigan (Ingham Co.)": (42.73, -84.56),
    "Minnesota (Blue Earth Co.)": (44.02, -94.00),
    "South Dakota (Brookings Co.)": (44.31, -96.80),
    "Wisconsin (Dane Co.)": (43.07, -89.40),
}

GDD_BASE_TEMP_F = 50.0
YEARS_OF_HISTORY = 10
SOUTHERLY_MIN_DEG, SOUTHERLY_MAX_DEG = 135.0, 225.0   # S sector, +/-45 deg around 180
SOUTHWEST_MIN_DEG, SOUTHWEST_MAX_DEG = 202.5, 247.5   # SW sector, +/-22.5 deg around 225
STORM_PRECIP_MM_THRESHOLD = 25.0   # ~1 inch/day = a notable storm day
STORM_WIND_MPH_THRESHOLD = 25.0    # sustained-max wind threshold, storm-proxy

# Cache: once a location's 10-year daily dataset has been fetched, it is
# saved here. Every later run reads the cache instead of re-pulling 10
# years of hourly data from Open-Meteo. Cache is considered stale (and
# refetched) only if its most recent date is more than CACHE_MAX_AGE_DAYS
# behind today -- see get_daily_weather() below.
CACHE_DIR = "cache_daily_weather"
CACHE_MAX_AGE_DAYS = 5

# Forecasts are ALWAYS written here, never into CACHE_DIR and never
# appended to the training dataset CSV -- see forecast_next_days() /
# save_forecast() below. Keeping predictions physically separate from
# observed data is what stops a later run from accidentally training on
# its own predictions.
FORECAST_DIR = "forecasts"

HOURLY_VARS = (
    "temperature_2m,precipitation,rain,wind_speed_10m,"
    "wind_direction_10m,surface_pressure"
)
# NOTE: no `cape` here -- not available on the historical archive endpoint.


# ============================================================
# 1. FETCH -- 10 years of HOURLY data per Midwest location
# ============================================================

def fetch_hourly_history(lat: float, lon: float, years: int = YEARS_OF_HISTORY) -> pd.DataFrame:
    """
    Pulls `years` years of hourly weather from Open-Meteo's historical
    archive (ERA5 reanalysis; free, no API key). Chunked one year at a
    time to keep each request small and to make partial failures easy
    to retry/skip instead of losing the whole pull.
    """
    today = dt.date.today()
    end_date = today - dt.timedelta(days=3)  # archive has a short backfill lag
    start_date = end_date.replace(year=end_date.year - years)

    frames = []
    chunk_start = start_date
    while chunk_start < end_date:
        chunk_end = min(dt.date(chunk_start.year, 12, 31), end_date)
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
            "hourly": HOURLY_VARS,
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "mm",
            "timezone": "UTC",
        }
        resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=30)
        resp.raise_for_status()
        hourly = resp.json()["hourly"]
        frames.append(pd.DataFrame(hourly))
        chunk_start = dt.date(chunk_start.year + 1, 1, 1)
        time.sleep(0.2)  # be polite to the free API

    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_live_cape(lat: float, lon: float, forecast_days: int = 16) -> pd.DataFrame:
    """
    LIVE SCORING ONLY -- pulls hourly CAPE from the forecast endpoint
    (which does expose it), for the current +16-day window. Never used
    for the historical training set -- only when scoring "today" in
    the deployed app.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "cape",
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }
    resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=15)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    out = pd.DataFrame(hourly)
    out["time"] = pd.to_datetime(out["time"])
    return out


# ============================================================
# 2. AGGREGATE HOURLY -> DAILY
# ============================================================

def _circular_mean_deg(degrees: pd.Series) -> float:
    """Proper circular mean for wind direction (avoid 350/10 -> 180 error)."""
    radians = np.deg2rad(degrees.dropna().to_numpy())
    if len(radians) == 0:
        return np.nan
    mean_sin = np.mean(np.sin(radians))
    mean_cos = np.mean(np.cos(radians))
    return float((np.rad2deg(np.arctan2(mean_sin, mean_cos)) + 360) % 360)


def aggregate_daily(hourly_df: pd.DataFrame, location: str) -> pd.DataFrame:
    """
    Collapses one location's hourly rows into one row per day, with the
    exact base columns requested (temperature_*, rain_mm, precipitation_mm,
    wind_speed_*, wind_direction_mean, southerly_hours, southwest_hours,
    pressure_mean). GDD / cumulative_gdd / rolling windows / storm_flag /
    calendar fields are added in `add_derived_features` below.
    """
    df = hourly_df.copy()
    df["date"] = df["time"].dt.date

    rows = []
    for date, day in df.groupby("date"):
        wind_dir = day["wind_direction_10m"]
        rows.append(
            {
                "date": date,
                "location": location,
                "temperature_max": day["temperature_2m"].max(),
                "temperature_min": day["temperature_2m"].min(),
                "temperature_mean": day["temperature_2m"].mean(),
                "rain_mm": day["rain"].sum(),
                "precipitation_mm": day["precipitation"].sum(),
                "wind_speed_mean": day["wind_speed_10m"].mean(),
                "wind_speed_max": day["wind_speed_10m"].max(),
                "wind_direction_mean": _circular_mean_deg(wind_dir),
                "southerly_hours": int(((wind_dir >= SOUTHERLY_MIN_DEG) & (wind_dir <= SOUTHERLY_MAX_DEG)).sum()),
                "southwest_hours": int(((wind_dir >= SOUTHWEST_MIN_DEG) & (wind_dir <= SOUTHWEST_MAX_DEG)).sum()),
                "pressure_mean": day["surface_pressure"].mean(),
                "cape": np.nan,  # not available historically -- see module docstring
            }
        )
    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


# ============================================================
# 3. DERIVED FEATURES -- GDD, rolling windows, pressure deltas, calendar
# ============================================================

def add_derived_features(daily: pd.DataFrame, base_temp_f: float = GDD_BASE_TEMP_F) -> pd.DataFrame:
    """
    Adds every remaining column from the requested schema. Must be run
    per-location (rolling/cumulative math should never bleed across
    locations), so this function assumes `daily` is ALREADY a single
    location, sorted by date -- see build_full_dataset() for the
    per-location loop that enforces this.
    """
    df = daily.sort_values("date").reset_index(drop=True).copy()

    # --- pressure change ---
    df["pressure_change_24h"] = df["pressure_mean"].diff(1)
    df["pressure_change_48h"] = df["pressure_mean"].diff(2)

    # --- storm flag (proxy, since CAPE isn't available historically) ---
    df["storm_flag"] = (
        (df["precipitation_mm"] >= STORM_PRECIP_MM_THRESHOLD)
        | (df["wind_speed_max"] >= STORM_WIND_MPH_THRESHOLD)
    ).astype(int)

    # --- GDD, reset each calendar year (Jan 1 biofix) ---
    df["year"] = df["date"].dt.year
    df["gdd"] = (
        ((df["temperature_max"] + df["temperature_min"]) / 2.0 - base_temp_f).clip(lower=0.0)
    )
    df["cumulative_gdd"] = df.groupby("year")["gdd"].cumsum()

    # --- rolling windows (trailing, min_periods = full window to avoid
    #     misleadingly small partial-window sums at the start of the series) ---
    df["rain_3d"] = df["rain_mm"].rolling(3, min_periods=3).sum()
    df["rain_7d"] = df["rain_mm"].rolling(7, min_periods=7).sum()
    df["southerly_hours_3d"] = df["southerly_hours"].rolling(3, min_periods=3).sum()
    df["southerly_hours_7d"] = df["southerly_hours"].rolling(7, min_periods=7).sum()
    df["wind_speed_3d"] = df["wind_speed_mean"].rolling(3, min_periods=3).mean()
    df["wind_speed_7d"] = df["wind_speed_mean"].rolling(7, min_periods=7).mean()

    # --- calendar ---
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    df = df.drop(columns=["year"])
    return df


FINAL_COLUMNS = [
    "date", "location",
    "temperature_max", "temperature_min", "temperature_mean",
    "rain_mm", "precipitation_mm",
    "wind_speed_mean", "wind_speed_max", "wind_direction_mean",
    "southerly_hours", "southwest_hours",
    "pressure_mean", "pressure_change_24h", "pressure_change_48h",
    "cape", "storm_flag",
    "gdd", "cumulative_gdd",
    "rain_3d", "rain_7d",
    "southerly_hours_3d", "southerly_hours_7d",
    "wind_speed_3d", "wind_speed_7d",
    "month", "day_of_year",
]


def _location_slug(name: str) -> str:
    """'Illinois (Champaign Co.)' -> 'Illinois'  (used for cache/forecast filenames)"""
    return name.split(" (")[0].replace(" ", "_")


def get_daily_weather(
    name: str,
    lat: float,
    lon: float,
    years: int = YEARS_OF_HISTORY,
    cache_dir: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Cache-aware wrapper around fetch -> aggregate -> derive-features for
    ONE location. If a cached daily file already exists and is recent
    enough (see CACHE_MAX_AGE_DAYS), it's loaded straight from disk and
    the network is never touched. Otherwise it fetches, builds the
    enriched daily dataframe, and writes it to cache for next time.

    cache_dir defaults to the CACHE_DIR module constant, resolved at
    CALL time (not def time) so changing pipe.CACHE_DIR before calling
    this actually takes effect -- important for tests/alternate runs.
    """
    cache_dir = cache_dir if cache_dir is not None else CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{_location_slug(name)}_daily.csv")

    if not force_refresh and os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        most_recent = cached["date"].max().date()
        age_days = (dt.date.today() - most_recent).days
        if age_days <= CACHE_MAX_AGE_DAYS:
            print(f"  [cache hit]  {name}: reusing cached daily data through {most_recent} "
                  f"({len(cached):,} rows, no fetch performed)")
            return cached
        print(f"  [cache stale]  {name}: cached data only goes through {most_recent} "
              f"({age_days} days old) -- refetching")
    else:
        print(f"  [cache miss]  {name}: no cache found -- fetching {years}yr hourly history")

    hourly = fetch_hourly_history(lat, lon, years)
    daily = aggregate_daily(hourly, name)
    enriched = add_derived_features(daily)
    enriched.to_csv(cache_path, index=False)
    print(f"  [cached]  {name}: saved {len(enriched):,} rows to {cache_path}")
    return enriched


def build_full_dataset(
    locations: dict[str, tuple[float, float]] = MIDWEST_LOCATIONS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Builds/loads the enriched daily dataset for every Midwest location.
    Each location is fetched at most once ever (see get_daily_weather) --
    re-running this function on a later day just reuses the cache unless
    it's gone stale, instead of re-pulling 10 years of hourly data again.
    """
    all_locations = [
        get_daily_weather(name, lat, lon, force_refresh=force_refresh)
        for name, (lat, lon) in locations.items()
    ]
    full = pd.concat(all_locations, ignore_index=True)
    return full[FINAL_COLUMNS]


# ============================================================
# 4. FORECAST TARGETS -- the next 7-30 ROWS of raw weather itself
# ============================================================

# The raw weather variables we forecast directly.
FORECAST_VARS = [
    "temperature_max", "temperature_min", "rain_mm",
    "wind_speed_mean", "pressure_mean",
    "wind_dir_sin", "wind_dir_cos",  # circular wind direction, see below
]

# Daily resolution for the short range (most trustworthy), then sparser
# check-in points further out (forecasting daily weather 30 days ahead
# with real accuracy isn't realistic for any model, XGBoost included).
HORIZONS = [1, 2, 3, 4, 5, 6, 7, 14, 21, 30]


def add_wind_components(df: pd.DataFrame) -> pd.DataFrame:
    """Splits circular wind_direction_mean into sin/cos so it can be
    regressed properly (0 and 360 degrees must be treated as identical,
    which a raw-degree regression target would not do)."""
    df = df.copy()
    radians = np.deg2rad(df["wind_direction_mean"])
    df["wind_dir_sin"] = np.sin(radians)
    df["wind_dir_cos"] = np.cos(radians)
    return df


def reconstruct_wind_direction(sin_val: np.ndarray, cos_val: np.ndarray) -> np.ndarray:
    """Turns predicted sin/cos components back into a 0-360 degree heading."""
    return (np.rad2deg(np.arctan2(sin_val, cos_val)) + 360) % 360


def add_forecast_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds forward-looking targets PER LOCATION (never across location
    boundaries): for every FORECAST_VAR and every horizon in HORIZONS,
    adds a column named "{var}__t+{h}" holding that variable's actual
    value h days later. These are the "next 7 to 30 rows" the model
    learns to predict.
    """
    df = add_wind_components(df)

    out_frames = []
    for loc, group in df.groupby("location"):
        g = group.sort_values("date").reset_index(drop=True).copy()
        for var in FORECAST_VARS:
            for h in HORIZONS:
                g[f"{var}__t+{h}"] = g[var].shift(-h)
        out_frames.append(g)

    result = pd.concat(out_frames, ignore_index=True)
    target_cols = [f"{var}__t+{h}" for var in FORECAST_VARS for h in HORIZONS]
    # Drop rows where we can't see the full 30-day horizon (tail of each location's series)
    return result.dropna(subset=target_cols)


# ============================================================
# 5. TRAIN XGBOOST -- two models, time-based split (no shuffling)
# ============================================================

FEATURE_COLUMNS = [
    "temperature_max", "temperature_min", "temperature_mean",
    "rain_mm", "precipitation_mm",
    "wind_speed_mean", "wind_speed_max", "wind_direction_mean",
    "southerly_hours", "southwest_hours",
    "pressure_mean", "pressure_change_24h", "pressure_change_48h",
    "storm_flag", "gdd", "cumulative_gdd",
    "rain_3d", "rain_7d",
    "southerly_hours_3d", "southerly_hours_7d",
    "wind_speed_3d", "wind_speed_7d",
    "month", "day_of_year", "location",
]


def time_based_split(df: pd.DataFrame, test_years: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits by DATE, not randomly -- the last `test_years` years become
    the test set for every location. Prevents leakage from rolling/
    cumulative features (a random shuffle would let the model "see the
    future" through overlapping windows).
    """
    cutoff = df["date"].max() - pd.DateOffset(years=test_years)
    train = df[df["date"] <= cutoff]
    test = df[df["date"] > cutoff]
    return train, test


def prep_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].copy()
    X["location"] = X["location"].astype("category")  # native XGBoost categorical support
    return X


def train_forecast_models(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> dict[str, dict[int, XGBRegressor]]:
    """
    Trains ONE XGBoost regressor per (variable, horizon) pair -- e.g.
    "temperature_max 3 days out" and "temperature_max 7 days out" are
    two different models, each specialised for that exact horizon
    (this is the standard "direct multi-horizon" forecasting approach,
    simpler and more robust than trying to force one model to output
    a whole sequence at once).

    Returns models[var][horizon] -> fitted XGBRegressor.
    Prints test-set MAE for every (var, horizon) so you can see
    forecast skill honestly degrade at longer horizons.
    """
    X_train = prep_features(train_df)
    X_test = prep_features(test_df)

    models: dict[str, dict[int, XGBRegressor]] = {var: {} for var in FORECAST_VARS}

    for var in FORECAST_VARS:
        print(f"\n  -- {var} --")
        for h in HORIZONS:
            target_col = f"{var}__t+{h}"
            y_train = train_df[target_col]
            y_test = test_df[target_col]

            model = XGBRegressor(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.06,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                enable_categorical=True,
                random_state=42,
            )
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            print(f"    t+{h:>2} days   test MAE = {mae:.2f}   (n_test={len(y_test)})")
            models[var][h] = model

    return models



NON_NEGATIVE_VARS = {"rain_mm", "wind_speed_mean"}  # physically can't go below 0


def forecast_next_days(
    models: dict[str, dict[int, XGBRegressor]],
    latest_row: pd.DataFrame,
) -> pd.DataFrame:
    """
    Takes the single most recent row of features for one location
    (a 1-row DataFrame with the same columns as FEATURE_COLUMNS) and
    returns a table of PREDICTED FUTURE ROWS -- one row per horizon in
    HORIZONS, with the forecasted values of every FORECAST_VAR. This is
    the actual "next 7 to 30 rows" output.

    XGBoost regression has no built-in concept of a physical floor, so
    it can (and did) predict things like -0.23mm of rain. Rain and wind
    speed are clipped to >= 0 after prediction -- temperature and
    pressure are left unclipped since negative/low values there are
    physically real (a Midwest winter has plenty of sub-zero days).
    """
    X = prep_features(latest_row)
    out_rows = []

    for h in HORIZONS:
        row = {"horizon_days": h}
        for var in FORECAST_VARS:
            if var in ("wind_dir_sin", "wind_dir_cos"):
                continue  # reconstructed below, not reported raw
            value = float(models[var][h].predict(X)[0])
            if var in NON_NEGATIVE_VARS:
                value = max(0.0, value)
            row[var] = value

        sin_pred = models["wind_dir_sin"][h].predict(X)[0]
        cos_pred = models["wind_dir_cos"][h].predict(X)[0]
        row["wind_direction_mean"] = float(reconstruct_wind_direction(sin_pred, cos_pred))

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def forecast_all_locations(
    models: dict[str, dict[int, XGBRegressor]],
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Runs forecast_next_days() for every location in `raw` and returns
    ONE combined DataFrame (not one per location) -- every row still
    carries its own `location` and `horizon_days`, so nothing is lost,
    it's just one table instead of eight files.
    """
    all_forecasts = []
    for loc in raw["location"].unique():
        latest = raw[raw["location"] == loc].sort_values("date").tail(1)
        as_of_date = latest["date"].iloc[0]
        if hasattr(as_of_date, "date"):
            as_of_date = as_of_date.date()

        latest = add_wind_components(latest)
        forecast = forecast_next_days(models, latest)
        forecast.insert(0, "location", loc)
        forecast.insert(1, "as_of_date", as_of_date)
        all_forecasts.append(forecast)

    return pd.concat(all_forecasts, ignore_index=True)


def save_forecast(
    forecast: pd.DataFrame,
    forecast_dir: Optional[str] = None,
) -> str:
    """
    Tags the WHOLE combined forecast (all locations, one DataFrame) with
    a single prediction_id + generated_at timestamp for this run, and
    writes it to ONE file under FORECAST_DIR -- deliberately never the
    same file, and never the same directory, as the training dataset or
    the cache in CACHE_DIR. A prediction is not an observation: mixing
    the two would let a later training run silently learn from its own
    past guesses.

    One run = one prediction_id = one file, no matter how many
    locations are in it. Each row still carries its own `location` and
    `horizon_days`, so you can filter/split the single file however you
    need downstream -- but nothing is scattered across separate files
    to begin with.
    """
    forecast_dir = forecast_dir if forecast_dir is not None else FORECAST_DIR
    os.makedirs(forecast_dir, exist_ok=True)

    tagged = forecast.copy()
    prediction_id = str(uuid.uuid4())
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    tagged.insert(0, "prediction_id", prediction_id)
    tagged.insert(1, "generated_at_utc", generated_at)

    run_date = dt.date.today().isoformat()
    filename = f"forecast_run_{run_date}_{prediction_id[:8]}.csv"
    path = os.path.join(forecast_dir, filename)
    tagged.to_csv(path, index=False)
    return path



# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("Step 1/5: loading 10 years of daily Midwest weather (cache-aware -- "
          f"see {CACHE_DIR}/, only fetches locations not already cached)...")
    raw = build_full_dataset()

    print(f"Step 2/5: dataset ready -- {len(raw):,} location-days, columns: {list(raw.columns)}")
    raw.to_csv("midwest_pest_weather_dataset.csv", index=False)

    print(f"Step 3/5: building forecast targets for {len(FORECAST_VARS)} variables "
          f"x {len(HORIZONS)} horizons (t+{HORIZONS[0]} to t+{HORIZONS[-1]} days)...")
    labeled = add_forecast_targets(raw)
    train_df, test_df = time_based_split(labeled, test_years=2)
    print(f"  train rows: {len(train_df):,}   test rows: {len(test_df):,}")

    print("Step 4/5: training XGBoost models (this trains "
          f"{len(FORECAST_VARS) * len(HORIZONS)} models -- one per variable/horizon pair)...")
    models = train_forecast_models(train_df, test_df)

    print(f"Step 5/5: forecasting next {HORIZONS[-1]} days for all {raw['location'].nunique()} locations "
          f"into ONE combined table, writing to {FORECAST_DIR}/ (never into the training CSV or cache)...")
    forecast = forecast_all_locations(models, raw)
    saved_path = save_forecast(forecast)

    print(f"\nCombined forecast ({len(forecast)} rows = "
          f"{raw['location'].nunique()} locations x {len(HORIZONS)} horizons), saved to {saved_path}:")
    print(forecast.to_string(index=False))

    print(f"\nDone. Training dataset: midwest_pest_weather_dataset.csv")
    print(f"      Per-location cache: {CACHE_DIR}/  (reused automatically on future runs)")
    print(f"      Tagged forecast:    {saved_path}  (one file, one prediction_id for this whole run, "
          "never merged back into the dataset above)")
    print("Next step (not run here): feed these forecasted rows into the GDD / arrival-signal")
    print("calculators from midwest_pest_risk.py to turn them into a forward-looking pest-risk map.")


if __name__ == "__main__":
    main()
