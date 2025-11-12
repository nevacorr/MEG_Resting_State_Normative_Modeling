import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def get_model_values(working_dir, data_dir, gender, n_splits):
    model_slope = pd.read_csv(os.path.join
                  (working_dir, os.path.join(working_dir, data_dir), f'{gender}_{n_splits}_splits_allsplits_slopes.csv'))

    model_ymin = pd.read_csv(os.path.join
                              (working_dir, os.path.join(working_dir, data_dir), f'{gender}_{n_splits}_splits_ymin.csv'))

    model_slope.rename(columns={'Unnamed: 0': 'band'}, inplace=True)
    model_ymin.rename(columns={'Unnamed: 0': 'band'}, inplace=True)

    model_slopes_dict = {band: band_df.drop(columns=['band', 'split']) for band, band_df in model_slope.groupby('band')}
    model_ymin_dict = {band: band_df.drop(columns=['band', 'split']) for band, band_df in model_ymin.groupby('band')}

    return model_slopes_dict, model_ymin_dict

def calculate_sig_of_change(model_slopes_dict, band):
    df_sig = pd.DataFrame()
    # Calculate confidence intervals
    for reg in model_slopes_dict[band].columns:
        slopes_reg = model_slopes_dict[band].loc[:, reg]
        lower_bound, upper_bound = np.percentile(slopes_reg.to_numpy(), [2.5, 97.5])
        if lower_bound < 0 < upper_bound:
            df_sig.loc[band, reg] = 0
        else:
            df_sig.loc[band, reg] = 1
    df_sig = df_sig.astype(int)
    return df_sig

def plot_model(gender, ages, age_conversion_factor, y_pred, band, region, pchange, slope, df_sig, model_slopes_dict):
    if gender == 'male':
        c = 'b'
    else:
        c = 'crimson'
    # plot model for this brain region
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(ages / age_conversion_factor, y_pred, c)
    # plt.ylim([0, 40])
    plt.title(
        f'Regions with Change in MEG power for\n {band} band in region {region}\n{gender} percent change = {pchange:.1f}\nslope = {slope * age_conversion_factor:.3f} sig change={df_sig.loc[band, region]}')
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    slopes_reg = model_slopes_dict[band].loc[:, region]
    lower_bound, upper_bound = np.percentile(slopes_reg.to_numpy(), [2.5, 97.5])
    plt.hist(slopes_reg * age_conversion_factor)
    plt.plot([lower_bound * age_conversion_factor, lower_bound * age_conversion_factor], [0, 20], 'r')
    plt.plot([upper_bound * age_conversion_factor, upper_bound * age_conversion_factor], [0, 20], 'r')
    plt.plot([0, 0], [0, 20], 'k')
    plt.title(
        f'{band} {gender} {region} sig = {df_sig.loc[band, region]}\nHistogram of slopes\nlower_bound = {lower_bound * age_conversion_factor:.2f} upper_bound = {upper_bound * age_conversion_factor:.2f}')
    plt.tight_layout()
    plt.show()
    # fname = os.path.join(working_dir, 'models_and_histograms', f'{band}_{gender}_{region}_plot_of_model_with_slope_and_histogram_of_slopes.png')
    # plt.savefig(fname)
    # plt.close()

    print(
        f'{band} {region} {gender} slope = {slope * age_conversion_factor: .3f} percent change = {pchange:.1f} sig change = {df_sig.loc[band, region]}')

def calculate_ypred_percent_change(band, region, agemin, agemax,  model_slopes_dict, model_ymin_dict, all_pred_lines, ages):
    slope = model_slopes_dict[band][region].mean()
    ymin = model_ymin_dict[band][region].mean()
    # Compute predicted values using the line equation
    y_pred = ymin + slope * (ages - agemin)
    all_pred_lines.append(y_pred)
    pchange = slope / ymin * (agemax - agemin) * 100.0
    return slope, y_pred, all_pred_lines, pchange