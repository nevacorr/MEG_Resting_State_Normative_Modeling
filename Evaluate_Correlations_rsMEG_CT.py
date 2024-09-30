###
# This programs evaluates correlations between Z-score values for regions showing accelerated cortical thinning at the
# post-COVID time point and regions showing reduced resting state MEG power at the post-COVID timepoint



for band in bands:
    Z_time2_male = pd.read_csv('{}/predict_files/{}_{}/Z_scores_by_region_postcovid_testset_Final.txt'
                               .format(working_dir, 'male', band))
    Z_time2_male.rename(columns={'subject_id_test': 'participant_id'}, inplace=True)

    Z_time2_female = pd.read_csv('{}/predict_files/{}_{}/Z_scores_by_region_postcovid_testset_Final.txt'
                                 .format(working_dir, 'female', band))
    Z_time2_female.rename(columns={'subject_id_test': 'participant_id'}, inplace=True)

    Z_time2[f'male_{band}'] = Z_time2_male
    Z_time2[f'female_{band}'] = Z_time2_female