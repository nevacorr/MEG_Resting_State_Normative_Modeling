import os
import pickle

working_dir = os.getcwd()
n_splits = 100
bands = ['beta', 'gamma']
metric_cols = ['MSLL', 'EV', 'SMSE', 'RMSE', 'Rho', 'z_mean', 'z_std']

blr_metrics = {}
blr_metrics_average = {}

for sex in ['male', 'female']:
    filepath = os.path.join(
        working_dir,
        f'blr_metrics_{sex}_{n_splits}_splits.pkl'
    )

    with open(filepath, 'rb') as f:
        blr_metrics[sex] = pickle.load(f)

    blr_metrics_average[sex] = {}

    for band in bands:
        df = blr_metrics[sex][band]

        mean_df = (
            df.groupby('ROI', as_index=False)[metric_cols]
            .mean()
        )

        blr_metrics_average[sex][band] = mean_df

        print(f'\n{sex} — {band}: average across splits')
        print(mean_df.to_string(index=False))