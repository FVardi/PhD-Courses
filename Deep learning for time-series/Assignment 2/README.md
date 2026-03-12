# London Smart Meters Dataset - Time Series Forecasting

## Project Overview

This project contains the UK Power Networks London Smart Meters dataset for energy consumption forecasting and time series analysis.

**Dataset**: London Smart Meters (UK Power Networks)
**Time Period**: 2012-10-12 to 2014-05-23
**Total Households**: 5,566
**Total Records**: ~177 million energy readings

---

## Directory Structure

```
Assignment 2/
├── README.md                              ← You are here
├── data/
│   ├── london_smart_meters/              ← Original dataset (source files)
│   │   ├── daily_dataset/
│   │   ├── halfhourly_dataset/
│   │   ├── hhblock_dataset/
│   │   ├── acorn_details.csv
│   │   ├── informations_households.csv
│   │   ├── weather_daily_darksky.csv
│   │   ├── weather_hourly_darksky.csv
│   │   └── uk_bank_holidays.csv
│   │
│   └── organized_output/                 ← Organized & documented version
│       ├── README.md
│       ├── QUICK_START.md
│       ├── DATA_SCHEMA_SPECIFICATION.json
│       ├── [reference CSV files]
│       └── [dataset folders with block indices]
└── scripts
```

## Key Column Identifications

### (i) Household Identifier Column

**Column Name**: `LCLid`
**Data Type**: String
**Format**: MAC###### (e.g., "MAC000002")
**Description**: Unique identifier for each smart meter / household
**Appears In**: All energy consumption files

### (ii) Timestamp Column

**For Halfhourly Data:**
- **Column Name**: `tstp`
- **Data Type**: datetime64[ns]
- **Format**: YYYY-MM-DD HH:MM:SS
- **Interval**: 30-minute intervals (00:00, 00:30, 01:00, 01:30, ..., 23:30)
- **File**: `halfhourly_dataset` blocks

**For Daily Data:**
- **Column Name**: `day`
- **Data Type**: datetime64[ns]
- **Format**: YYYY-MM-DD
- **Interval**: Daily (24-hour)
- **Files**: `daily_dataset` and `hhblock_dataset` blocks

### (iii) Consumption Target Column

**For Halfhourly Data:**
- **Column Name**: `energy(kWh/hh)`
- **Data Type**: float64
- **Units**: kWh (kilowatt-hours)
- **Description**: Energy consumption per 30-minute period
- **Range**: >= 0
- **File**: `halfhourly_dataset` blocks

**For Daily Aggregated Data:**
- **Column Name**: `energy_sum`
- **Data Type**: f1t64
- **Units**: kWh
- **Description**: Total daily energy consumption
- **Range**: >= 0
- **File**: `daily_dataset` blocks
- **Alternative Statistics**: energy_median, energy_mean, energy_max, energy_min, energy_std, energy_count

**For Daily Pivot Format (48 half-hourly columns):**
- **Column Names**: `hh_0`, `hh_1`, `hh_2`, ..., `hh_47`
- **Data Type**: float64
- **Units**: kWh
- **Description**: Energy consumption for each half-hourly period of the day
- **Range**: >= 0
- **Note**: hh_0 = 00:00-00:30, hh_1 = 00:30-01:00, ..., hh_47 = 23:30-24:00
- **File**: `hhblock_dataset` blocks

### (iv) Household Meta Labels

**ACORN Classification** (From `informations_households.csv`):
- **Column Name**: `Acorn`
- **Data Type**: String
- **Categories**: 20 types (ACORN-A through ACORN-Q)
- **Description**: Demographic and socioeconomic classification
- **Examples**:
  - ACORN-A: Affluent achievers
  - ACORN-Q: Council residents
- **Join Key**: `LCLid`
- **File**: `informations_households.csv`

**ACORN Grouped** (From `informations_households.csv`):
- **Column Name**: `Acorn_grouped`
- **Data Type**: String
- **Description**: Higher-level ACORN grouping
- **Join Key**: `LCLid`
- **File**: `informations_households.csv`

**Pricing plan** (From `informations_households.csv`):
- **Column Name**: `stdornt`
- **Data Type**: String
- **Description**: Pricing plan of the household (dynamic time-of-use of standard flat rate price)
- **Join Key**: `LCLid`
- **File**: `informations_households.csv`

---

## Data Files and File Paths

### Energy Consumption Datasets

#### 1. Daily Dataset (Daily Aggregated Statistics)
**Location**: `data/london_smart_meters/daily_dataset/daily_dataset/`

**Files**: `block_0.csv` through `block_111.csv` (112 blocks total)

**Schema**:
```
LCLid (string)
day (datetime)
energy_median (float64)
energy_mean (float64)
energy_max (float64)
energy_min (float64)
energy_sum (float64)
energy_std (float64)
energy_count (int64)
```

**Sample File**: `data/london_smart_meters/daily_dataset/daily_dataset/block_0.csv`
**Total Records**: ~3.5 million
**Block Size**: 25,000-36,000 rows per block
**File Size per Block**: ~2.5-3.5 MB

---

#### 2. Halfhourly Dataset (Fine-Grained 30-minute readings)
**Location**: `data/london_smart_meters/halfhourly_dataset/halfhourly_dataset/`

**Files**: `block_0.csv` through `block_111.csv` (112 blocks total)

**Schema**:
```
LCLid (string)
tstp (datetime)
energy(kWh/hh) (float64)
```

**Sample File**: `data/london_smart_meters/halfhourly_dataset/halfhourly_dataset/block_0.csv`
**Total Records**: ~170 million
**Block Size**: 21,000-40,000 rows per block
**File Size per Block**: ~50-70 MB (⚠️ Large files)

---

#### 3. HHBlock Dataset (Daily view with 48 half-hourly columns)
**Location**: `data/london_smart_meters/hhblock_dataset/hhblock_dataset/`

**Files**: `block_0.csv` through `block_111.csv` (112 blocks total)

**Schema**:
```
LCLid (string)
day (datetime)
hh_0 (float64)
hh_1 (float64)
...
hh_47 (float64)
```

**Sample File**: `data/london_smart_meters/hhblock_dataset/hhblock_dataset/block_0.csv`
**Total Records**: ~3.5 million
**Block Size**: 26,000-35,000 rows per block
**File Size per Block**: ~11-15 MB

---

### Reference/Metadata Files

#### ACORN Details (Demographic Categories)
**File Path**: `data/london_smart_meters/acorn_details.csv`
**Rows**: 826
**Columns**: 20 (one for each ACORN category: ACORN-A through ACORN-Q)
**Purpose**: Demographic and socioeconomic attribute values for each ACORN category
**File Size**: 0.12 MB

---

#### Household Information (Household Metadata)
**File Path**: `data/london_smart_meters/informations_households.csv`
**Rows**: 5,566 (one per household)
**Columns**: 5
```
LCLid (household identifier - PRIMARY KEY)
stdornt (standard of urbanization - urban/rural)
stdornt_deployment (urbanization details)
Acorn (ACORN demographic category)
Acorn_grouped (ACORN higher-level grouping)
file (source file reference)
```
**Purpose**: Link household IDs to demographic classifications
**File Size**: 0.22 MB
**Key Usage**: Join with energy data on `LCLid` to add household labels

---

#### Weather Data - Daily
**File Path**: `data/london_smart_meters/weather_daily_darksky.csv`
**Rows**: 882
**Columns**: 32 (including date, temperature, humidity, cloudCover, windSpeed, etc.)
**Purpose**: Daily weather features for forecasting
**File Size**: 0.33 MB

---

#### Weather Data - Hourly
**File Path**: `data/london_smart_meters/weather_hourly_darksky.csv`
**Rows**: 21,165
**Columns**: 12 (including time, temperature, humidity, cloudCover, windSpeed, etc.)
**Purpose**: Hourly weather features for sub-daily forecasting
**File Size**: 1.94 MB

---

#### UK Bank Holidays
**File Path**: `data/london_smart_meters/uk_bank_holidays.csv`
**Rows**: 25
**Columns**: 2 (date, holiday name)
**Purpose**: Calendar features for forecasting (flag special days)
**File Size**: 0.76 KB

---
