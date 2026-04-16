# Task 01: Load and Validate Schema

## Schema Specifications

### Household Identifier Column
- **Column Name**: `LCLid`
- **Data Type**: String (object)
- **Format**: MAC###### (e.g., "MAC000002")
- **Purpose**: Unique meter/household identifier
- **Used For**: Sorting, grouping, joining with metadata

### Timestamp Columns
**Halfhourly Format**:
- **Column Name**: `tstp`
- **Format**: YYYY-MM-DD HH:MM:SS
- **Interval**: 30-minute readings (00:00, 00:30, 01:00, ...)
- **Type**: datetime64[ns]

**Daily Format**:
- **Column Name**: `day`
- **Format**: YYYY-MM-DD
- **Interval**: One entry per household per day
- **Type**: datetime64[ns]

### Energy Columns
**Halfhourly**:
- Column: `energy(kWh/hh)` - float64, units: kWh

**Daily Aggregated**:
- `energy_sum` - Total daily consumption
- `energy_mean` - Mean consumption
- `energy_median` - Median consumption
- `energy_max` - Peak consumption
- `energy_min` - Minimum consumption
- `energy_std` - Standard deviation
- `energy_count` - Number of half-hourly readings

**Daily Pivot Format**:
- Columns: `hh_0` to `hh_47` - 48 half-hourly consumption values
- Each represents 30-minute period's consumption

### Optional Household Metadata
- **Tarriff type**: Standard or Time of Use rate
- **ACORN Affluence**: Category in format ACORN-X
