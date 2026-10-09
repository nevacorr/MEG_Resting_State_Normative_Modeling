import pandas as pd

# =========================================================
# FILES
# =========================================================

male_file = "output_data/male_20_splits_BLR_LOO_metrics.csv"
female_file = "output_data/female_20_splits_BLR_LOO_metrics.csv"

# =========================================================
# READ FILES
# =========================================================

male = pd.read_csv(male_file)
female = pd.read_csv(female_file)

male["sex"] = "Male"
female["sex"] = "Female"

df = pd.concat([male, female], ignore_index=True)

# =========================================================
# AVERAGE METRICS BY SEX, BAND, AND BRAIN REGION
# =========================================================

summary = (
    df.groupby(["sex", "band", "ROI"])[
        ["EV", "RMSE", "z_mean", "z_std"]
    ]
    .mean()
    .reset_index()
)

# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\nAverage BLR LOO metrics by brain region:")
print(summary.to_string(index=False))

# =========================================================
# SAVE RESULTS
# =========================================================

output_file = "BLR_LOO_average_metrics_by_region.csv"
summary.to_csv(output_file, index=False)

print(f"\nSaved to: {output_file}")

