import pandas as pd
import matplotlib.pyplot as plt

filename = '/home/toddr/neva/PycharmProjects/data_dir/genz_rs_power_rel_vfix_alln_December2024.csv'

resting_state_data = pd.read_csv(filename)

df = resting_state_data.drop(columns=['id', 'gender', 't1_age', 't2_age'])

bands = ['alpha', 'theta', 'beta', 'gamma']

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for row, timepoint in enumerate(['t1', 't2']):
    for col, band in enumerate(bands):
        ax = axes[row, col]

        # Select columns for this timepoint and band
        cols = [c for c in df.columns
                if c.startswith(f'{timepoint}_{band}-')]

        # Region names
        regions = [c.replace(f'{timepoint}_{band}-', '') for c in cols]

        # One line per subject
        for _, subject in df[cols].iterrows():
            ax.plot(regions, subject.values, alpha=0.3)

        ax.set_title(f'{timepoint.upper()} {band}')
        ax.tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()
mystop=1