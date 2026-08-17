# 🛒 Walmart Retail Goods Forecasting

> **Hierarchical demand forecasting for Walmart retail goods** — predicting 28 days of daily unit sales across thousands of products, stores, and states using a gradient-boosted **LightGBM** model trained on the [M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) dataset.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white">
  <img alt="LightGBM" src="https://img.shields.io/badge/Model-LightGBM-2ea44f">
  <img alt="Pandas" src="https://img.shields.io/badge/Data-pandas%20%7C%20numpy-150458?logo=pandas&logoColor=white">
  <img alt="Task" src="https://img.shields.io/badge/Task-Time%20Series%20Forecasting-orange">
  <img alt="Horizon" src="https://img.shields.io/badge/Horizon-28%20days-lightgrey">
</p>

**Author:** Syed Zain Raza

---

## 📑 Table of Contents

- [Overview](#-overview)
- [The Problem](#-the-problem)
- [Dataset](#-dataset)
- [Repository Structure](#-repository-structure)
- [Methodology](#-methodology)
  - [1. Exploratory Data Analysis](#1-exploratory-data-analysis)
  - [2. Data Preparation](#2-data-preparation)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. Modeling](#4-modeling)
  - [5. Recursive Forecasting & Submission](#5-recursive-forecasting--submission)
- [Model Details](#-model-details)
- [Getting Started](#-getting-started)
- [Results & Interpretation](#-results--interpretation)
- [Key Design Choices](#-key-design-choices)
- [Possible Improvements](#-possible-improvements)
- [Acknowledgements](#-acknowledgements)

---

## 🔎 Overview

Retailers live or die by how well they anticipate demand. Over-forecast and shelves overflow with capital tied up in unsold inventory; under-forecast and you lose sales to empty shelves. This project builds an end-to-end machine-learning pipeline that forecasts the **daily unit sales of individual Walmart products for the next 28 days**.

The solution treats forecasting as a **supervised regression problem**: historical sales are reshaped into a long, tidy table where each row is a *(product, store, day)* observation, enriched with calendar, pricing, and lag-based features, and a **LightGBM** regressor is trained to predict units sold. Predictions are generated **recursively**, one day at a time, so that each new day's forecast can feed the lag features of the following day.

Everything — the exploration, the feature pipeline, the model training, and the submission logic — lives in a single, well-commented Jupyter notebook: [`walmart_M5_SZR_Github.ipynb`](walmart_M5_SZR_Github.ipynb).

---

## 🎯 The Problem

This project addresses the **M5 Forecasting – Accuracy** challenge:

> Given **5+ years** of daily unit-sales history for **3,049 products** sold across **10 Walmart stores** in **3 US states** (California, Texas, Wisconsin), forecast the unit sales of each product for the following **28 days**.

The products span **3 categories** (Hobbies, Household, Foods) and **7 departments**, forming a natural **hierarchy** (item → department → category → store → state → total). Sales are **highly intermittent** — many items sell zero units on most days — which makes this a genuinely hard forecasting problem that classical smoothing methods struggle with.

---

## 📦 Dataset

The pipeline expects the raw M5 competition files in a `forecast_walmart_data/` directory at the repository root:

```
forecast_walmart_data/
├── calendar.csv                 # Date → day mapping, weekday/month/year, events, SNAP flags
├── sales_train_validation.csv   # Daily unit sales per item/store (d_1 … d_1913)
├── sell_prices.csv              # Weekly selling price per item/store
└── sample_submission.csv        # Submission template (28-day forecast horizon)
```

> ⚠️ **The raw CSVs are not included in this repository** (they are large and governed by Kaggle's competition rules). Download them from the [M5 Forecasting – Accuracy data page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and place them in `forecast_walmart_data/`.

| File | What it contains | Key columns |
|------|------------------|-------------|
| `calendar.csv` | One row per calendar day | `date`, `wm_yr_wk`, `weekday`, `wday`, `month`, `year`, `event_name_1/2`, `event_type_1/2`, `snap_CA/TX/WI` |
| `sales_train_validation.csv` | Wide table of daily sales | `id`, `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`, `d_1 … d_1913` |
| `sell_prices.csv` | Weekly prices | `store_id`, `item_id`, `wm_yr_wk`, `sell_price` |
| `sample_submission.csv` | Forecast template | `id`, `F1 … F28` |

**Scale of the data:** ~30K item/store series × ~1,900 days ≈ **58 million** potential observations, aggressively down-cast to `int16`/`float32` to keep memory in check.

---

## 🗂 Repository Structure

```
walmart_RetailGoods_Forecasting/
├── walmart_M5_SZR_Github.ipynb   # 📓 End-to-end notebook: EDA → features → training → submission
├── model_new.lgb                 # 💾 Pre-trained LightGBM model (ready to load & predict)
├── README.md                     # 📄 You are here
└── forecast_walmart_data/        # 📥 (You provide) Raw M5 CSVs — see Dataset section
```

---

## 🧪 Methodology

The notebook flows through five clear stages.

### 1. Exploratory Data Analysis

Before modeling, the notebook builds intuition about the data:

- **Single-item time series** — plots a product's raw daily sales (`HOBBIES_1_369_CA_1`), revealing long flat stretches where the item simply wasn't sold.
- **Calendar-aligned view** — merges the item series with real dates for interpretable trends.
- **Seasonality breakdowns** — average sales by **day of week**, **month**, and **year**, exposing weekly and annual seasonality.
- **Category composition** — counts of items per category (Hobbies / Household / Foods).
- **Category-level sales** — aggregate demand curves per category over time.
- **Store-level demand** — a **180-day rolling average** of total sales for each of the 10 stores, making long-run store trends comparable.

### 2. Data Preparation

The wide sales table is reshaped into the modeling format via `merge_Cal_Sell(...)`, which:

1. Loads sales, calendar, and price data with **memory-efficient dtypes** (`int16`, `float32`, `category`).
2. Encodes categorical IDs as compact integer codes.
3. **Melts** the wide `d_1 … d_1913` columns into a long *(id, day, sales)* format.
4. **Merges** in calendar attributes (on `d`) and selling prices (on `store_id`, `item_id`, `wm_yr_wk`).
5. In inference mode, appends 28 empty future days (`d_1914 … d_1941`) to be filled by the model.

### 3. Feature Engineering

`feature_Creation(...)` enriches every row with signals that let a tree model capture temporal structure:

| Feature family | Features | Why it matters |
|----------------|----------|----------------|
| **Lag features** | `lag_7`, `lag_28` | Sales from 1 and 4 weeks ago — the strongest predictors of near-term demand |
| **Rolling means** | `rmean_7_7`, `rmean_7_28`, `rmean_28_7`, `rmean_28_28` | Smoothed recent trend over 7- and 28-day windows on top of each lag |
| **Calendar** | `wday`, `week`, `month`, `quarter`, `year`, `mday` | Encodes weekly / monthly / yearly seasonality |
| **Events** | `event_name_1/2`, `event_type_1/2` | Holidays and special events that spike or suppress demand |
| **SNAP** | `snap_CA`, `snap_TX`, `snap_WI` | Days when food-assistance benefits boost grocery sales |
| **Pricing** | `sell_price` | Price sensitivity of demand |
| **Identity** | `item_id`, `dept_id`, `store_id`, `cat_id`, `state_id` | Lets the model learn product- and location-specific baselines |

Rows with `NaN`s introduced by the lag/rolling windows are dropped before training.

### 4. Modeling

A single global **LightGBM** model is trained across all series:

- **2,000,000** randomly sampled rows are held out as a validation set (seeded for reproducibility); the remainder form the training set.
- Data is wrapped in `lgb.Dataset` objects with the categorical features declared explicitly so LightGBM uses its native categorical splits.
- The model optimizes a **Poisson objective** — the natural choice for non-negative count data like unit sales — and is evaluated with **RMSE**.

### 5. Recursive Forecasting & Submission

Because lag features depend on recent sales, the 28-day horizon is predicted **one day at a time**: each predicted day is written back into the series so it can populate the lags of the next day. To add robustness, the notebook produces an **ensemble of predictions scaled by multiplicative factors** (`alpha ≈ 1.033 / 1.027 / 1.021`) — a common M5 "magic multiplier" trick that nudges forecasts upward to counter the model's tendency to under-predict spiky demand — and averages them into the final submission.

---

## ⚙️ Model Details

The trained model is saved as [`model_new.lgb`](model_new.lgb) and can be loaded directly without re-running the (memory- and time-intensive) training step.

| Hyperparameter | Value |
|----------------|-------|
| `objective` | `poisson` |
| `metric` | `rmse` |
| `learning_rate` | `0.095` |
| `num_iterations` | `3200` |
| `num_leaves` | `90` |
| `min_data_in_leaf` | `25` |
| `sub_row` (bagging fraction) | `0.75` |
| `bagging_freq` | `1` |
| `lambda_l2` | `0.2` |
| `force_row_wise` | `True` |
| `early_stopping_round` | `1` |

**Model features (25 total):** `item_id`, `dept_id`, `store_id`, `cat_id`, `state_id`, `wday`, `month`, `year`, `event_name_1`, `event_type_1`, `event_name_2`, `event_type_2`, `snap_CA`, `snap_TX`, `snap_WI`, `sell_price`, `lag_7`, `lag_28`, `rmean_7_7`, `rmean_7_28`, `rmean_28_7`, `rmean_28_28`, `week`, `quarter`, `mday`.

Load and use the pre-trained model:

```python
import lightgbm as lgb

model = lgb.Booster(model_file="model_new.lgb")
# predictions = model.predict(X)   # X must contain the 25 features above, in order
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook / JupyterLab

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ZainRaza14/walmart_RetailGoods_Forecasting.git
cd walmart_RetailGoods_Forecasting

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn lightgbm jupyter

# 3. Download the M5 dataset from Kaggle and place the CSVs here:
#    forecast_walmart_data/{calendar,sales_train_validation,sell_prices,sample_submission}.csv

# 4. Launch the notebook
jupyter notebook walmart_M5_SZR_Github.ipynb
```

### Running the pipeline

Execute the notebook cells top to bottom to:

1. Explore the data (EDA plots).
2. Build the training frame and engineer features.
3. Train the LightGBM model (or skip this and load `model_new.lgb`).
4. Generate the recursive 28-day forecast and write the submission CSV.

> 💡 **Tip:** Training on the full dataset is memory-hungry. The pipeline already down-casts dtypes and frees intermediate frames with `gc.collect()`. If you're memory-constrained, reduce the training window via the `f_Day` argument to `merge_Cal_Sell(...)` or lower the number of sampled rows.

---

## 📈 Results & Interpretation

- The model produces a **28-day-ahead forecast** for every item/store series, formatted to the M5 submission spec (`id`, `F1 … F28`).
- The **Poisson objective** respects the count nature of the target and prevents negative forecasts.
- **Lag and rolling-mean features dominate** predictive power — recent demand is the best predictor of near-future demand — while calendar, event, SNAP, and price features let the model modulate that baseline around holidays, paydays, and promotions.
- The **alpha-multiplier ensemble** compensates for the systematic under-prediction that global count models exhibit on intermittent, spiky series.

---

## 🧠 Key Design Choices

- **A single global model** instead of thousands of per-series models — it shares statistical strength across products and generalizes to sparse items that lack enough history to model alone.
- **Gradient-boosted trees over deep learning** — LightGBM handles heterogeneous tabular features, native categoricals, and intermittent targets efficiently, with far less tuning and compute than a neural forecaster.
- **Recursive multi-step forecasting** — keeps a single model responsible for the whole horizon while honoring the autoregressive lag structure.
- **Aggressive memory management** — `int16`/`float32` down-casting, `category` codes, and explicit garbage collection make a 50M+ row problem tractable on modest hardware.

---

## 🔮 Possible Improvements

- Add **price-change and rolling-price** features (relative price, price momentum).
- Introduce **more lags/windows** (e.g. `lag_1`, `lag_14`, 90-day rolling stats).
- Train **per-store or per-category models** and blend with the global model.
- Optimize directly for the competition's **WRMSSE** metric with hierarchical weighting.
- Add **cross-validation with time-based splits** for more honest error estimates.
- Package the pipeline into reusable `.py` modules with a CLI.

---

## 🙏 Acknowledgements

- **Dataset:** [M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) competition, hosted by Kaggle in partnership with Walmart and the University of Nicosia.
- **Core libraries:** [LightGBM](https://lightgbm.readthedocs.io/), [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/).

---

<p align="center"><i>Built for accurate, memory-efficient, large-scale retail demand forecasting.</i></p>
