import glob
import os
import pandas as pd
import numpy as np


# --------------------------------------------------
# 1. Find latest forecast
# --------------------------------------------------

forecast_files = sorted(
    glob.glob("forecasts/forecast_run_*.csv")
)

if not forecast_files:
    raise FileNotFoundError("No forecast files found.")

input_file = forecast_files[-1]

print(f"Using forecast: {input_file}")


# --------------------------------------------------
# 2. Load forecast
# --------------------------------------------------

df = pd.read_csv(input_file)


# --------------------------------------------------
# 3. Create What-If copy
# --------------------------------------------------

whatif = df.copy()

rng = np.random.default_rng()


# --------------------------------------------------
# 4. Randomize weather values
# --------------------------------------------------

# Temperature
whatif["temperature_max"] = (
    whatif["temperature_max"]
    + rng.uniform(-8, 8, len(whatif))
)

whatif["temperature_min"] = (
    whatif["temperature_min"]
    + rng.uniform(-6, 6, len(whatif))
)


# Rain
# MUST stay below 4.0 mm
whatif["rain_mm"] = rng.uniform(
    0,
    2.5,
    len(whatif)
)


# Wind speed
# MUST stay below 10 mph
whatif["wind_speed_mean"] = rng.uniform(
    3,
    8,
    len(whatif)
)


# Pressure
whatif["pressure_mean"] = (
    whatif["pressure_mean"]
    + rng.uniform(-8, 8, len(whatif))
)


# --------------------------------------------------
# 5. Keep Tmax >= Tmin
# --------------------------------------------------

mask = whatif["temperature_max"] < whatif["temperature_min"]

whatif.loc[mask, [
    "temperature_max",
    "temperature_min"
]] = whatif.loc[mask, [
    "temperature_min",
    "temperature_max"
]].values


# --------------------------------------------------
# 6. Save
# --------------------------------------------------

os.makedirs("whatif", exist_ok=True)

output_file = "whatif/whatif.csv"

whatif.to_csv(
    output_file,
    index=False
)

print(f"What-If forecast saved to: {output_file}")


# --------------------------------------------------
# 7. Verify gates
# --------------------------------------------------

# Only check the first 2 forecast horizons
next_48h = (
    whatif
    .groupby("horizon_days")
    .agg({
        "rain_mm": "mean",
        "wind_speed_mean": "mean"
    })
    .sort_index()
    .head(2)
)

rain_48h = next_48h["rain_mm"].sum()
wind_mean_48h = next_48h["wind_speed_mean"].mean()

rain_gate = rain_48h > 4.0
wind_gate = wind_mean_48h > 10.0

print("\n--- WHAT-IF WEATHER GATES ---")
print(
    f"Rain Gate: {rain_48h:.2f} > 4.0 = {rain_gate}"
)
print(
    f"Wind Gate: {wind_mean_48h:.2f} > 10.0 = {wind_gate}"
)
