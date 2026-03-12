
# %%
import pandas as pd
# %%

path = 'C:/Users/au808956/Documents/Repos/PhD-Courses/Deep learning for time-series/Assignment 2/data/london_smart_meters/'
df = pd.read_csv(path + "halfhourly_dataset/halfhourly_dataset/block_9.csv", dtype={"energy(kWh/hh)": "float"}, na_values="Null")

# %%
df["energy(kWh/hh)"].apply(type).unique()

