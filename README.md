# Canadian Rent & Mortgage Forecasting

This repository forecasts Canadian apartment rents and mortgage loans using Statistics Canada data. The models capture how rent pressure is driven by demand (population growth, migration), supply constraints (housing starts with lag), and interest rates. The mortgage module shows rate transmission and credit conditions.

## Headline Metrics

**Rent Model Performance** (Rolling backtest: 13 quarters, 1,032 predictions):

| Model | MAE (CAD) | sMAPE (%) |
|-------|-----------|-----------|
| **ElasticNet** | 42.34 | **2.61%** |
| Lag-1 baseline | 44.00 | 2.74% |
| Lag-4 baseline | 117.30 | 7.48% |

**Uplift vs Lag-4 baseline**: 63.9% MAE improvement, 65.1% sMAPE improvement

The ElasticNet model outperforms both naive baselines, with particularly strong gains over the seasonal (Lag-4) baseline, demonstrating the value of incorporating exogenous features (population, migration, housing starts, interest rates).

## Toronto & GTA Focus

This project focuses on Toronto and the Greater Toronto Area (GTA) for detailed analysis and visualization. **Important note**: The data uses Census Metropolitan Area (CMA) boundaries. Cities like Mississauga, Brampton, Oakville, and Burlington are part of the Toronto CMA, so they appear as "Toronto" in the dataset rather than as separate municipalities.

## The Story

**Rent Model**: Rent pressure comes from three main forces:
- **Demand**: Population growth and international migration increase housing demand
- **Supply lag**: Housing starts take time to become available units, creating supply constraints
- **Rates**: Interest rates affect both rental demand and investment in rental properties

**Mortgage Model**: Tracks how interest rates transmit through the financial system to mortgage lending, showing credit conditions and borrowing patterns.

## Data Sources

All data comes from Statistics Canada CSV extracts saved locally in `data/raw/`:

| File | StatCan Table ID | Description | Frequency |
|------|------------------|-------------|-----------|
| `4610009201-noSymbol.csv` | 46-10-0092-01 | Asking rent prices | Quarterly |
| `1010013401-noSymbol.csv` | 10-10-0134-01 | Chartered banks mortgage loans | Quarterly |
| `3810023801-noSymbol.csv` | 38-10-0238-01 | Household credit market summary | Quarterly |
| `1010014501-noSymbol.csv` | 10-10-0145-01 | Financial market statistics | Weekly |
| `1410028701-noSymbol.csv` | 14-10-0287-01 | Labour force characteristics | Monthly |
| `1410032001-noSymbol.csv` | 14-10-0320-01 | Monthly employees count | Monthly |
| `34100156.csv` | 34-10-0156 | Housing starts | Monthly |
| `17100009.csv` | 17-10-0009 | Population estimates | Quarterly |
| `17100040.csv` | 17-10-0040 | International migration components | Quarterly |
| `17100121.csv` | 17-10-0121 | Non-permanent residents | Quarterly |

## Aggregation Rules

All data is aggregated to quarterly for modeling:

- **Quarterly data**: Use quarter-end Timestamp
- **Weekly rates**: Take last observation in quarter (quarter-end value)
- **Monthly labour/employment**: Use quarterly average
- **Monthly housing starts**: Use quarterly sum
- **Quarterly population/migration**: Use directly

## Models

**Rent Model (Elastic Net)**: Forecasts asking rents for 1-bedroom and 2-bedroom apartments by CMA. Uses lagged rents, exogenous features (population, migration, starts, rates), and seasonality.

**Mortgage Model (SARIMAX)**: Forecasts chartered bank mortgage loans at Canada level. Uses SARIMAX with exogenous regressors (rates, credit conditions, employment).

## Running the Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python run_pipeline.py
```

The pipeline will:
1. Create output directories
2. Preprocess targets (rent and mortgage)
3. Build exogenous features
4. Create modeling datasets
5. Train and backtest both models
6. Generate plots

## Outputs

All outputs are saved to `outputs/`:

**Predictions**:
- `rent_predictions.csv`: Rent forecasts with actual vs predicted
- `mortgage_predictions.csv`: Mortgage forecasts with actual vs predicted

**Metrics**:
- `rent_metrics.json`: MAE and sMAPE overall and by unit type
- `mortgage_metrics.json`: MAE and sMAPE for mortgage forecasts

**Model Details**:
- `rent_model_coefficients.csv`: Elastic Net feature coefficients
- `mortgage_model_summary.txt`: SARIMAX model summary

**Plots** (in `outputs/plots/`):
- `rent_overall.png`: Overall rent forecast
- `rent_by_unit_type.png`: Rent forecast by unit type
- `rent_top_cmas.png`: Rent forecast for top 6 CMAs
- `mortgage_actual_vs_predicted.png`: Mortgage forecast

**Processed Data** (in `data/processed/`):
- `mortgage_target_quarterly.parquet`: Mortgage target series
- `rent_target_quarterly.parquet`: Rent target series (if available)
- `exog_quarterly.parquet`: Exogenous features
- `rent_model_dataset.parquet`: Rent modeling dataset
- `mortgage_model_dataset.parquet`: Mortgage modeling dataset

## Evaluation

Models are evaluated using rolling backtest (expanding window):
- Train on all data up to time t
- Forecast t+1 (and optionally t+2)
- Calculate MAE and sMAPE metrics

**Backtest Coverage**: 13 quarters (Q3 2022 - Q3 2025), 1,032 predictions across 41 CMAs and 2 unit types.

Metrics are reported overall and broken down by unit type (for rent) and forecast horizon.

## Q1 2026 Consensus Outlook (Base Case)

This outlook is based on model outputs, observed trends through 2025Q3, and structural relationships in the data. It represents a qualitative base case, not point forecasts.

### Rates (Bank Rate / Yields)

**Base Case:**
- Bank rate likely holds or eases modestly from current levels, conditional on inflation data
- 5-year GoC yield reflects market expectations for gradual easing; remains sensitive to inflation surprises
- Key drivers: inflation persistence, labour market tightness, housing market stability
- Rate transmission to mortgages operates with 1-2 quarter lag

**Upside Case:** Faster disinflation allows earlier cuts, supporting mortgage growth and rental demand

**Downside Case:** Inflation re-accelerates or housing stress intensifies, forcing higher-for-longer policy

### Mortgages (Mortgage Loans Outstanding Growth)

**Base Case:**
- Growth likely remains subdued or slightly negative in Q1 2026, reflecting rate sensitivity and lagged effects
- Housing demand proxies (population growth, migration) remain supportive but rate headwinds persist
- Credit conditions tighten further if rates stay elevated; loosening if cuts materialize
- SARIMAX model shows strong rate sensitivity with 1-2 quarter transmission lag

**Upside Case:** Rate cuts materialize, unlocking pent-up demand and accelerating mortgage growth

**Downside Case:** Prolonged high rates or housing market stress leads to further credit contraction

### Rent (Toronto + GTA Focus)

**Base Case:**
- 1-bedroom asking rent: moderate upward pressure continues, driven by population growth and migration inflows
- 2-bedroom asking rent: similar direction but potentially less pressure than 1-bedroom (supply/demand balance)
- Housing starts lag effects: recent starts will add supply with 2-3 quarter delay, but demand growth may outpace
- Rate effects: higher rates reduce investor demand for rental properties but also reduce homeownership, increasing rental demand
- ElasticNet model shows strong signals from migration and population growth; rate effects are mixed

**Upside Case:** Migration surge continues, housing starts lag, rates ease (supporting demand), leading to stronger rent growth

**Downside Case:** Migration slows, housing starts accelerate, rates stay high (reducing demand), leading to flat or declining rents

**Important Limitations:**
- Data covers CMA boundaries (Toronto CMA includes Mississauga, Brampton, etc.)
- This is asking rent (new listings), not existing tenant rent (which adjusts more slowly)
- Regional variations within GTA are not captured by CMA-level aggregation

## Key Figures and What They Show

### Rent Forecasts: Toronto & GTA

![Toronto 1-Bedroom Forecast](figures/rent_toronto_1bed_forecast.png)

**Toronto 1-Bedroom Rent Forecast**
- Shows actual vs predicted asking rent (CAD) for Toronto CMA 1-bedroom apartments, quarterly from 2019Q1 through 2025Q3
- Includes baseline comparisons (Lag-1 and Lag-4) to show model value-add
- Focus on: how well the ElasticNet model tracks actual rent movements, especially during rate cycles and migration surges
- Note: This is asking rent (new listings), not existing tenant rent. CMA boundaries include Mississauga, Brampton, etc.

![GTA 1-Bedroom Forecast](figures/rent_gta_1bed_forecast.png)

**GTA Proxy 1-Bedroom Rent Forecast**
- Aggregates Toronto + Oshawa CMAs (mean) to approximate Greater Toronto Area dynamics
- Same structure as Toronto chart: actual vs predicted vs baselines, quarterly
- Focus on: regional trends beyond core Toronto, capturing broader GTA rental pressure
- Note: This is a proxy aggregate; true GTA would require more granular CMA data

### Model Performance Analysis

![Uplift vs Lag-4 Heatmap](figures/rent_uplift_lag4_heatmap.png)

**Model Improvement vs Seasonal Baseline (Heatmap)**
- Shows sMAPE improvement (%) for ElasticNet vs Lag-4 baseline, by CMA and unit type
- Color scale: green = large improvement, red/yellow = smaller improvement
- Focus on: which markets and unit types benefit most from exogenous features (population, migration, rates, housing starts)
- Note: Negative values (if any) indicate rare cases where the seasonal baseline outperformed; most cells show positive uplift

![Top 15 CMAs by Improvement](figures/rent_uplift_top15_cmas.png)

**Top 15 CMAs by Model Improvement**
- Ranks CMAs by average uplift (across unit types) vs Lag-4 baseline
- Horizontal bar chart showing sMAPE improvement percentage
- Focus on: markets where demand/supply dynamics are most predictable from exogenous features
- Note: Higher uplift doesn't necessarily mean lower absolute error; it means the model adds more value relative to naive seasonality

### Rate Story

![Toronto QoQ vs 5-Year GoC Yield](figures/toronto_rent_qoq_vs_goc5y.png)

**Toronto Rent QoQ Change vs 5-Year GoC Yield**
- Dual-axis plot: rent quarter-over-quarter change (%) vs 5-year Government of Canada bond yield (%)
- Quarterly data aligned to period-end dates
- Focus on: short-horizon rent momentum and its relationship to medium-term rate expectations
- Note: QoQ changes are volatile; this shows correlation, not causation. The 5-year yield reflects market expectations for medium-term rates.

![GTA Proxy QoQ vs 5-Year GoC Yield](figures/gta_rent_qoq_vs_goc5y.png)

**GTA Proxy Rent QoQ Change vs 5-Year GoC Yield**
- Same structure as Toronto chart, but aggregates Toronto + Oshawa CMAs
- Focus on: whether broader GTA shows similar rate sensitivity patterns
- Note: Proxy aggregate may smooth some local volatility

### Mortgage Forecast

![Mortgage Actual vs Predicted](figures/mortgage_actual_vs_predicted.png)

**Chartered Bank Mortgage Loans Outstanding**
- Shows actual vs predicted mortgage loans outstanding (CAD billions) at Canada level, quarterly
- SARIMAX model with exogenous regressors (rates, credit conditions, employment)
- Focus on: how well the model captures rate transmission effects and credit cycle dynamics
- Note: This is aggregate Canada-level data; regional mortgage patterns may differ

