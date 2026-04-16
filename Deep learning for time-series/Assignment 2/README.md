# Assignment 2 — Machine Learning for Time Series

## Data Source

London Smart Meters dataset (provided course data bundle). Daily aggregates of half-hourly household electricity consumption readings.

**Note on granularity:** The provided bundle contains the *daily* version of the dataset (one row per household per day, aggregating up to 48 half-hourly slots). The assignment targets half-hourly granularity; all tasks use daily resolution accordingly (e.g. weekly seasonality at lag 7 days instead of 336 half-hours).

---

## File Paths

| File | Description |
|------|-------------|
| `data/london_smart_meters/daily_dataset/daily_dataset/block_0.csv` … `block_111.csv` | Raw consumption data (112 block files, ~50 households each) |
| `data/london_smart_meters/informations_households.csv` | Household metadata (Acorn group, tariff type) |
| `data/london_smart_meters/weather_daily_darksky.csv` | Daily weather data |
| `data/london_smart_meters/weather_hourly_darksky.csv` | Hourly weather data |
| `data/london_smart_meters/uk_bank_holidays.csv` | UK bank holidays |
| `data/processed/full_dataset.parquet` | All blocks merged + metadata joined |
| `data/processed/train.parquet` | Training split |
| `data/processed/val.parquet` | Validation split |
| `data/processed/test.parquet` | Test split |

---

## Column Names

### Consumption data (block files / processed dataset)

| Column | Role | Description |
|--------|------|-------------|
| `LCLid` | **household_id** | Unique household identifier |
| `day` | **timestamp** | Date of reading (daily resolution, parsed as datetime) |
| `energy_sum` | **target** | Total kWh consumed per day |
| `energy_mean` | feature | Mean kWh per half-hour slot (energy_sum / energy_count) |
| `energy_median` | feature | Median kWh per half-hour slot |
| `energy_std` | feature | Std dev of half-hourly consumption within the day |
| `energy_min` | feature | Min half-hourly consumption within the day |
| `energy_max` | feature | Max half-hourly consumption within the day |
| `energy_count` | feature | Number of valid half-hourly readings (max 48) |

### Household metadata (informations_households.csv)

| Column | Role | Description |
|--------|------|-------------|
| `LCLid` | join key | Household identifier (matches consumption data) |
| `Acorn_grouped` | **meta_label** | Socio-economic group: Affluent / Comfortable / Adversity / ACORN-U |
| `Acorn` | meta | Fine-grained Acorn category (e.g. ACORN-A through ACORN-Q) |
| `stdorToU` | meta | Tariff type: `Std` (standard) or `ToU` (time-of-use) |
| `file` | — | Source block file (not used downstream) |

---

## Experimental Split

| Split | Period | Rows |
|-------|--------|------|
| Train | before 2014-01-01 | 3,212,334 |
| Validation | 2014-01-01 – 2014-01-31 | 157,488 |
| Test | 2014-02-01 – 2014-02-28 | 140,611 |

The prescribed boundaries match the available data exactly (raw data spans 2011-11-23 to 2014-02-28).

---

## Setup

Run the setup script to load all blocks, validate schema, join metadata, apply the split, and write processed parquet files:

```bash
python scripts/setup.py
```
