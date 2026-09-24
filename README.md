# AIS-37 [PLACEHOLDER README NOT FINAL]
AIS-37 : Early Crop Disease &amp; Pest Outbreak Prediction. UN SDGs: SDG 2, SDG 12, SDG 13, SDG 15  

# PS-2A: Early Crop Disease & Pest Outbreak Prediction
 
**Track:** Track 2 — Smart Agriculture
**Status:** Placeholder — build in progress
 
## Overview
 
An image recognition + weather-integrated system to catch crop disease early
and flag pest outbreak risk before it spreads, so farmers spray less and
smarter instead of blanket-treating every field.
 
Two independent components:
 
1. **Leaf Disease Classifier** — CNN (transfer learning) trained on
   PlantVillage, identifies crop disease from a leaf photo.
2. **Regional Pest Risk Score** — weather-driven risk model (temperature /
   humidity / rainfall thresholds) estimating outbreak likelihood for a
   given region and time window.
Outputs are shown side by side rather than mathematically fused, to keep
each piece demoable on its own.
 
## Aligned UN SDGs
 
- **SDG 2 — Zero Hunger**: protects crop yield and food supply by catching
  disease/pest pressure early.
- **SDG 12 — Responsible Consumption & Production**: cuts unnecessary
  chemical pesticide use by targeting only what's actually needed.
- **SDG 13 — Climate Action**: pest risk model accounts for the shifting,
  climate-driven conditions (warmer winters, changed rainfall) that are
  expanding pest ranges.
- **SDG 15 — Life on Land**: reduces pesticide runoff into soil and nearby
  water/land ecosystems.
## Regional Focus: US Midwest
 
Demo and validation are scoped to the US Midwest (corn/soybean belt),
because:
 
- PlantVillage already covers corn and soybean disease classes.
- Climate research documents real, ongoing northward pest range shifts
  into the Midwest (e.g. corn leafhopper, corn earworm), driven by milder
  winters that let overwintering pests survive further north than before.
- Public reference data exists for the region — the Midwest Suction Trap
  Network (aphid/leafhopper migration monitoring across 8 Midwest states)
  and EDDMapS occurrence data — to ground the risk model in real patterns
  rather than an invented rule set.
- Underlying architecture is not region-locked — swapping coordinates and
  crop type generalizes it beyond the Midwest.
## Data Sources
 
| Type | Source |
|---|---|
| Leaf images (primary) | PlantVillage (Kaggle) |
| Leaf images (complementary) | PlantDoc, IP102 |
| Weather | Open-Meteo API / NASA POWER |
| Pest reference | Midwest Suction Trap Network, EDDMapS |
 
## TODO
 
- [ ] Fine-tune image classifier on PlantVillage
- [ ] Build weather-based risk scoring logic
- [ ] Wire up live demo UI
- [ ] Add Grad-CAM explainability overlay
- [ ] Pesticide-reduction impact calculator
 
