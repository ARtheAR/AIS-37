import glob
import pandas as pd
import numpy as np
import json

print("Select forecast mode:")
print("1. Prediction")
print("2. What-If Scenario")

choice = input("Enter 1 or 2: ").strip()


# --------------------------------------------------
# Load selected forecast
# --------------------------------------------------

if choice == "1":

    # Load latest combined forecast
    forecast_files = sorted(
        glob.glob("forecasts/forecast_run_*.csv")
    )

    if not forecast_files:
        raise FileNotFoundError(
            "No forecast files found in forecasts/"
        )

    df = pd.read_csv(forecast_files[-1])

    print(f"Loaded prediction: {forecast_files[-1]}")


elif choice == "2":

    forecast_files = "whatif/whatif.csv"

    try:
        df = pd.read_csv(forecast_files)
    except FileNotFoundError:
        raise FileNotFoundError(
            "No What-If forecast found at whatif/whatif.csv"
        )

    print(f"Loaded What-If scenario: {forecast_files}")


else:
    raise ValueError("Invalid choice. Please enter 1 or 2.")

# Focus on near-term critical window (Days 1-7) across locations (or pick a specific location)
# e.g., evaluating Illinois (Champaign Co.) or the regional average
eval_df = df[df["horizon_days"] <= 7].copy()

# Date / Phenology context
as_of_date = pd.to_datetime(eval_df["as_of_date"].iloc[0])
month = as_of_date.month

# Weather aggregations
t_max_mean = eval_df["temperature_max"].mean()
t_min_mean = eval_df["temperature_min"].mean()
t_mean = (eval_df["temperature_max"] + eval_df["temperature_min"]) / 2.0
rain_sum = eval_df["rain_mm"].clip(lower=0.0).sum()
wind_speed_mean = eval_df["wind_speed_mean"].mean()

# Check for Southerly / Southwesterly migration wind conditions (135° - 247.5°)
southerly_days = eval_df[
    (eval_df["wind_direction_mean"].between(135.0, 247.5)) & 
    (eval_df["wind_speed_mean"] >= 7.0)
]
has_migration_corridor = len(southerly_days) >= 3

# Compute GDD (Base 50°F) over the 7-day forecast
daily_gdd = np.maximum(0, ((eval_df["temperature_max"] + eval_df["temperature_min"]) / 2.0) - 50.0)
accumulated_gdd = daily_gdd.sum()

# -------------------------------------------------------------
# 1. Black cutworm (Mid-March – early June | Storm fronts)
# -------------------------------------------------------------
if 3 <= month <= 6:
    if has_migration_corridor and rain_sum >= 15.0:
        bcw_score = 5
    elif has_migration_corridor or rain_sum >= 10.0:
        bcw_score = 3
    else:
        bcw_score = 2
else:
    # September is completely outside the migration/cutting phenological window
    bcw_score = 1

# -------------------------------------------------------------
# 2. Corn earworm (Mid-July – September | Storm fronts / South winds)
# -------------------------------------------------------------
if 7 <= month <= 9:
    if has_migration_corridor and t_min_mean >= 55.0:
        # Favorable wind corridor + warm overnight temps = active moth influx
        cew_score = 4 if rain_sum >= 5.0 else 3
    elif has_migration_corridor:
        cew_score = 3
    else:
        cew_score = 2
else:
    cew_score = 1

# -------------------------------------------------------------
# 3. Potato leafhopper (May – June | Weather patterns + temp)
# -------------------------------------------------------------
if 5 <= month <= 6:
    if has_migration_corridor and t_max_mean >= 75.0:
        plh_score = 4
    else:
        plh_score = 3
elif month in [7, 8]:
    plh_score = 2
else:
    # September cold nights and rain suppress flight and population growth
    plh_score = 1

# -------------------------------------------------------------
# 4. Soybean aphid (June onward | Local buildup based on GDD & Temp)
# -------------------------------------------------------------
# Optimal reproduction: 72°F - 82°F; ceases < 50°F
avg_temp = t_mean.mean()
if 6 <= month <= 9:
    if 70.0 <= avg_temp <= 82.0 and accumulated_gdd >= 80.0:
        sba_score = 4
    elif accumulated_gdd >= 50.0:
        sba_score = 3
    else:
        sba_score = 2
else:
    sba_score = 1

# Output format
print(f"Black cutworm - {bcw_score}")
print(f"Corn earworm - {cew_score}")
print(f"Potato leafhopper - {plh_score}")
print(f"Soybean aphid - {sba_score} (calculated based on gdd)")



#file 2

risk_scores = {
    "Black cutworm": bcw_score,
    "Corn earworm": cew_score,
    "Potato leafhopper": plh_score,
    "Soybean aphid": sba_score
}

# Sort by forecast day
df = df.sort_values("horizon_days")

# Average weather variables across all locations
midwest_forecast = (
    df.groupby("horizon_days")
      .agg({
          "temperature_max": "mean",
          "temperature_min": "mean",    
          "rain_mm": "mean",
          "wind_speed_mean": "mean",
          "pressure_mean": "mean",
          "wind_direction_mean": "mean"
      })
      .reset_index()
)

next_48h = midwest_forecast.head(2)

rain_48h = next_48h["rain_mm"].sum()

wind_mean_48h = (
    next_48h["wind_speed_mean"].mean()
)


rain_gate = rain_48h > 4.0
wind_gate = wind_mean_48h > 10.0

# =========================
# 6. PRINT MIDWEST WEATHER
# =========================

print("\n========================================")
print("MIDWEST PEST RISK & WEATHER DECISION")
print("========================================")

print(f"Forecast file: {forecast_files[-1]}")
print(
    f"Locations averaged: "
    f"{df['location'].nunique()}"
)

print("\n--- MIDWEST WEATHER ---")

print(
    f"Average rain next 24–48h: "
    f"{rain_48h:.2f} mm"
)

print(
    f"Average wind next 24–48h: "
    f"{wind_mean_48h:.2f} mph"
)


# =========================
# 7. WEATHER CONDITIONS
# =========================

print("\n--- WEATHER GATES ---")

print(
    f"Rain Gate "
    f"(rain_mm > 4.0 mm): "
    f"{rain_48h:.2f} > 4.0 = {rain_gate}"
)

print(
    f"Wind Gate "
    f"(wind_speed_mean > 10 mph): "
    f"{wind_mean_48h:.2f} > 10 = {wind_gate}"
)


# =========================
# 8. PEST DECISIONS
# =========================

print("\n--- PEST DECISIONS ---")

for pest_name, score in risk_scores.items():

    print(f"\n{pest_name}")
    print(f"Risk score: {score}")

    # Zero-spray rule
    zero_spray = score <= 2

    print(
        f"Score <= 2 "
        f"(Zero-Spray Rule): "
        f"{score} <= 2 = {zero_spray}"
    )

    if zero_spray:
        print(
            "Decision: WITHHOLD CHEMICAL APPLICATION"
        )
        continue


    # Targeted intervention rule
    targeted_intervention = score >= 4

    print(
        f"Score >= 4 "
        f"(Targeted Intervention Rule): "
        f"{score} >= 4 = {targeted_intervention}"
    )

    if targeted_intervention:

        # Rain gate
        print(
            f"Rain Gate: {rain_gate}"
        )

        if rain_gate:
            print(
                "Rain condition TRUE: "
                "BLOCK FOLIAR LIQUID SPRAY"
            )
            print(
                "Alternative: delay application "
                "or use banded soil granules"
            )
        else:
            print(
                "Rain condition FALSE: "
                "Foliar liquid spray is not blocked "
                "by rain."
            )


        # Wind gate
        print(
            f"Wind Gate: {wind_gate}"
        )

        if wind_gate:
            print(
                "Wind condition TRUE: "
                "PROHIBIT AERIAL/FINE-DROPLET SPRAYING"
            )
        else:
            print(
                "Wind condition FALSE: "
                "Aerial/fine-droplet spraying is not "
                "blocked by wind."
            )


    # Score = 3
    if score == 3:
        print(
            "Decision: MONITOR / NO AUTOMATIC "
            "TARGETED INTERVENTION"
        )


#file 3


OUTCOMES = {
    "no_imminent_risk": "No imminent risk: WITHHOLD PESTICIDES",
    "minimal_risk": "Minimal risk: WITHHOLD OR CONTROLLED",
    "moderate_risk": "Moderate risk: MONITOR / CONTROLLED INTERVENTION",
    "high_risk": "High risk: TARGETED INTERVENTION",
    "severe_risk": "Severe risk: TARGETED INTERVENTION",

    "rain_gate": "High precipitation: BLOCK FOLIAR LIQUID SPRAY",
    "wind_gate": "High wind: PROHIBIT AERIAL / FINE-DROPLET SPRAYING",
    "rain_and_wind_gate": "High precipitation AND high wind: BLOCK FOLIAR LIQUID SPRAY AND PROHIBIT AERIAL / FINE-DROPLET SPRAYING"
}


def get_risk_outcome(score):
    if score <= 1:
        return OUTCOMES["no_imminent_risk"]
    elif score == 2:
        return OUTCOMES["minimal_risk"]
    elif score == 3:
        return OUTCOMES["moderate_risk"]
    elif score == 4:
        return OUTCOMES["high_risk"]
    else:
        return OUTCOMES["severe_risk"]


if rain_gate and wind_gate:
    environmental_override = OUTCOMES["rain_and_wind_gate"]

elif rain_gate:
    environmental_override = OUTCOMES["rain_gate"]

elif wind_gate:
    environmental_override = OUTCOMES["wind_gate"]

else:
    environmental_override = None


risk_results = {}

for pest_name, score in risk_scores.items():

    if environmental_override is not None:
        proposal = environmental_override
    else:
        proposal = get_risk_outcome(score)

    risk_results[pest_name] = {
        "score": score,
        "proposal": proposal
    }



risk_scores = {
    "Black cutworm": bcw_score,
    "Corn earworm": cew_score,
    "Potato leafhopper": plh_score,
    "Soybean aphid": sba_score
}


risk_results = {}

for pest_name, score in risk_scores.items():

    if environmental_override is not None:
        proposal = environmental_override
    else:
        proposal = get_risk_outcome(score)

    risk_results[pest_name] = {
        "score": score,
        "proposal": proposal
    }


with open("risk_scores.json", "w") as f:
    json.dump(risk_results, f, indent=4)
