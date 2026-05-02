# Multi-Location Temporal Forecasting of Temperature and Relative Humidity Using a Stacked Ensemble Framework

## Overview
This repository contains the full implementation of the study:

**“Multi-Location Temporal Forecasting of Near-Surface Temperature and Relative Humidity Across Diverse Climatic Regions of India Using a Stacked Ensemble Framework.”**

The project develops a machine learning framework for forecasting hourly temperature and relative humidity across multiple Indian cities using a stacked ensemble approach.

---

## Key Features
- Multi-city meteorological forecasting framework  
- Stacked ensemble model (XGBoost, LightGBM, CatBoost, HistGradientBoosting)  
- Multi-output regression (temperature and humidity prediction)  
- Walk-forward validation for time-series robustness  
- Cross-location generalisation (Leave-One-City-Out validation)  
- SHAP-based interpretability analysis  

---

## Dataset
The dataset used in this study is publicly available:

Indian Weather Repository – Live  
https://www.kaggle.com/code/nelgiriyewithana/indian-weather-repository-live  

---

## Repository Structure

The repository is organised as follows:

- `analysis/`  
  Contains the main analysis scripts, model development, and forecasting workflow.

- `archive/`  
  Includes earlier versions of the modelling scripts retained for transparency, version tracking, and reproducibility.

- `Charts/`  
  Visualisations and output plots generated during the analysis and forecasting process.

- `ne_110m_admin_0_countries/`  
  Geospatial data used for mapping and visual representation.

- `IndianWeatherRepository.csv`  
  Primary dataset used for model training and evaluation.

- `*.png` files  
  Forecast outputs and visual results.

- `*.html` files  
  Interactive visualisations (e.g., maps).

- `README.md`  
  Documentation describing the project, methodology, and usage instructions.

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/Bilqis1424/IndianWeatherCities.git
cd IndianWeatherCities
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the model:
```bash
python india_weather_rev6.py
```

