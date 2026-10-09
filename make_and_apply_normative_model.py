import pandas as pd
import numpy as np
import os
from helper_functions_MEG import plot_num_subjs
from helper_functions_MEG import recreate_folder
from prepare_rsMEG_data import prepare_rsMEG_data
from sklearn.model_selection import StratifiedShuffleSplit
from make_model import make_model
import pickle

def make_and_apply_normative_model(struct_var, show_plots, show_nsubject_plots, spline_order,
                               spline_knots, working_dir, ct_data_dir, MEG_filename,
                               subjects_to_exclude, bands, n_splits, data_type, evaluate_model_fit):

    # load all rs MEG data
    # returns all data for v1 and v2, list of unique subjects with MEG data at any time point, list of subjects with data from v1 only, list of subjects with data from v2 only
    rsd_v1, rsd_v2, all_subjects_orig, sub_v1_only_orig, sub_v2_only_orig \
                                 = prepare_rsMEG_data(MEG_filename, subjects_to_exclude, ct_data_dir)


    # Remove the prefix 't1_' or 't2_' from column names
    rsd_v1.columns = rsd_v1.columns.str.replace(r'^t1_', '', regex=True)
    rsd_v2.columns = rsd_v2.columns.str.replace(r'^t2_', '', regex=True)

    # Scale response variables
    cols_to_eval = [col for col in rsd_v1.columns if '-lh' in col or '-rh' in col]

    if data_type == 'relative':
        # # Multiply values in the specified columns by 100 (relative data)
        rsd_v1[cols_to_eval] = rsd_v1[cols_to_eval] * 100.000
        rsd_v2[cols_to_eval] = rsd_v2[cols_to_eval] * 100.000
    elif data_type == 'absolute':
        # Divide values in the specified columns by 100 (absolute data)
        rsd_v1[cols_to_eval] = rsd_v1[cols_to_eval] / 100.000
        rsd_v2[cols_to_eval] = rsd_v2[cols_to_eval] / 100.000

     # show bar plots with number of subjects per age group in pre-COVID data
    if show_nsubject_plots:

        plot_num_subjs(rsd_v1, f'Subjects by Age with Pre-COVID MEGrs Data\n'
                                 '(Total N=' + str(rsd_v1.shape[0]) + ')', struct_var, 'pre-covid_allsubj',
                                  os.path.join(working_dir, 'data'))

    sub_v1_only = sub_v1_only_orig
    sub_v2_only = sub_v2_only_orig

    all_subjects = rsd_v1['subject'].tolist()
    all_subjects.extend(rsd_v2['subject'].tolist())
    all_subjects = pd.unique(all_subjects).tolist()
    # all subjects of this with data at any timepoint
    all_subjects.sort()
    # all subjects with data only at both timepoints
    all_subjects_2ts = [sub for sub in all_subjects if (sub not in sub_v1_only and sub not in sub_v2_only)]

    # all time 1 data
    v1_df_for_train_test_split = rsd_v1.copy()

    # Create a dataframe that has only visit 1 data and only subject number, visit, age and sex as columns
    cols_to_keep = ['subject', 'gender', 'agegrp', 'agedays']
    cols_to_drop = [col for col in v1_df_for_train_test_split if col not in cols_to_keep]
    v1_df_for_train_test_split.drop(columns=cols_to_drop, inplace=True)
    # keep only the subjects that have data at both time points
    v1_df_for_train_test_split = v1_df_for_train_test_split[
        v1_df_for_train_test_split['subject'].isin(all_subjects_2ts)]

    # Initialize StratifiedShuffleSplit for equal train/test sizes. Use split percent that gives equal train and
    # test sizes for M and F after taking into account number of subjects with data only at one timepoint.
    # For males there are 47 subj. with data at both tp, 25 with data at tp 1 and 6 with data at tp 2 (from split add 14 at tp1 and 33 at tp2)
    # For females there are 53 subj. with data at both tp, 21 with data at tp 1 and 5 with data at tp 2 (from split add 18 at tp1 and 35 at tp2)
    # For males solve 25+x = 6+y and x+y= 47. get x=14, y=33 and test_size=33/47=0.7. This give 39 in train set and 39 in test set.
    # For females Solve 21+x = 5+y  and x+y = 53. get x=18, y=35 and test_size=35/53=0.66. This gives 39 in train set and 40 in test set


    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.70, random_state=42)
    # elif gender == "female":
    #     splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.66, random_state=42)

    strata = (v1_df_for_train_test_split['agegrp'].astype(str) + '_' + v1_df_for_train_test_split['gender'].astype(str))

    train_set_list = []
    test_set_list = []
    # Perform the splits
    for i, (train_index, test_index) in enumerate(
            splitter.split(v1_df_for_train_test_split, strata)):
        train_set_list_tmp = v1_df_for_train_test_split.iloc[train_index, 0].values.tolist()
        # add subjects with time 1 data only
        train_set_list_tmp.extend(sub_v1_only)
        test_set_list_tmp = v1_df_for_train_test_split.iloc[test_index, 0].values.tolist()
        # add subjects with time 2 data only
        test_set_list_tmp.extend(sub_v2_only)
        train_set_list.append(train_set_list_tmp)
        test_set_list.append(test_set_list_tmp)

    train_set_array = np.array(list(train_set_list))
    test_set_array = np.array(list(test_set_list))

    fname_train = '{}/visit1_subjects_train_sets_{}_splits_{}.npy'.format(working_dir, n_splits, struct_var)
    np.save(fname_train, train_set_array)

    fname_test = '{}/visit1_subjects_test_sets_{}_splits_{}.npy'.format(working_dir, n_splits, struct_var)
    np.save(fname_test, test_set_array)

    Z2_all_splits_dict = make_model(rsd_v1, rsd_v2, struct_var, n_splits, train_set_array, test_set_array,
               show_nsubject_plots, working_dir, spline_order, spline_knots, show_plots, bands, evaluate_model_fit)

    if evaluate_model_fit:
        return None

    # For each band, average Z scores for the same subject across splits
    for band, df in Z2_all_splits_dict.items():
        # Drop the 'split' column if it exists
        if 'split' in df.columns:
            df = df.drop(columns='split')

        # Select the region columns
        region_cols = [col for col in df.columns if '-lh' in col or '-rh' in col]

        # Group by subject_id and average only region columns
        averaged_df = df.groupby('subject_id_test')[region_cols].mean().reset_index()

        # Store the result back in the dictionary
        Z2_all_splits_dict[band] = averaged_df

        with open(os.path.join(working_dir, f'Zscores_post_covid_test_all_bands_{n_splits}_splits.pkl'), 'wb') as f:
            pickle.dump(Z2_all_splits_dict, f)

    return Z2_all_splits_dict