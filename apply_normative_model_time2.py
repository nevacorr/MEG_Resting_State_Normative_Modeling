#####
# This program imports the model and Z-scores from the bayesian linear regression normative modeling of the
# training data set (which is the adolescent visit 1 data). It then uses the model to calculate Z-scores for
# the post-covid adolescent (visit 2) data.
# Author: Neva M. Corrigan
# Date: 21 February, 2024
######
import os
import pandas as pd
from matplotlib import pyplot as plt
from Load_Genz_Data import load_genz_data
from plot_num_subjs import plot_num_subjs
from Utility_Functions import makenewdir, movefiles, create_dummy_design_matrix_one_gender
from Utility_Functions import plot_data_with_spline_one_gender, create_design_matrix_one_gender, read_ages_from_file
import shutil
from normative_edited import predict

def apply_normative_model_time2(gender, orig_struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                                orig_data_dir, working_dir):

    ######################## Apply Normative Model to Post-Covid Data ############################

    # load all brain and behavior data for visit 2
    visit = 2
    brain_good, all_data, roi_ids = load_genz_data(orig_struct_var, visit, orig_data_dir)

    #load brain and behavior data for visit 1
    visit = 1
    brain_v1, all_v1, roi_v1 = load_genz_data(orig_struct_var, visit, orig_data_dir)

    #extract subject numbers from visit 1 and find subjects in visit 2 that aren't in visit 1
    subjects_visit1 = all_v1['participant_id']
    rows_in_v2_but_not_v1 = all_data[~all_data['participant_id'].isin(all_v1['participant_id'])].dropna()
    subjs_in_v2_not_v1 = rows_in_v2_but_not_v1['participant_id'].copy()
    subjs_in_v2_not_v1 = subjs_in_v2_not_v1.astype(int)
    #only keep subjects at 12, 14 and 16 years of age (subject numbers <400) because cannot model 18 and 20 year olds
    subjs_in_v2_not_v1 = subjs_in_v2_not_v1[subjs_in_v2_not_v1 < 400]

    if gender == 'male':
        # keep only data for males
        all_data = all_data.loc[all_data['sex'] == 1]
        struct_var = 'cortthick_male'
    else:
        # keep only data for females
        all_data = all_data.loc[all_data['sex'] == 2]
        struct_var = 'cortthick_female'

    #remove sex column
    all_data = all_data.drop(columns=['sex'])

    #only include subjects that were not in the training set
    fname='{}/visit1_subjects_excluded_from_normative_model_test_set_{}_9_11_13.txt'.format(orig_data_dir, orig_struct_var)
    subjects_to_include = pd.read_csv(fname, header=None)
    subjects_to_include = pd.concat([subjects_to_include, subjs_in_v2_not_v1])
    brain_good = brain_good[brain_good['participant_id'].isin(subjects_to_include[0])]
    all_data = all_data[all_data['participant_id'].isin(subjects_to_include[0])]

    #write subject numbers used in test set to file
    subjects_test = all_data['participant_id'].tolist()
    fname = '{}/visit2_all_subjects_used_in_test_set_{}_{}.txt'.format(working_dir, struct_var, gender)
    file1 = open(fname, "w")
    for subj in subjects_test:
        file1.write(str(subj) + "\n")
    file1.close()

    #make file diretories for output
    makenewdir('{}/predict_files/'.format(working_dir))
    makenewdir('{}/predict_files/{}'.format(working_dir, struct_var))
    makenewdir('{}/predict_files/{}/plots'.format(working_dir, struct_var))
    makenewdir('{}/predict_files/{}/ROI_models'.format(working_dir, struct_var))
    makenewdir('{}/predict_files/{}/covariate_files'.format(working_dir, struct_var))
    makenewdir('{}/predict_files/{}/response_files'.format(working_dir, struct_var))

    # reset indices
    brain_good.reset_index(inplace=True)
    all_data.reset_index(inplace=True, drop=True)
    #read agemin and agemax from file
    agemin, agemax = read_ages_from_file(working_dir, struct_var)

    #show number of subjects by gender and age
    if gender == "female":
        genstring = 'Female'
    elif gender == "male":
        genstring = 'Male'
    if show_nsubject_plots:
        plot_num_subjs(all_data, gender, f'{genstring} Subjects with Post-COVID Data\nEvaluated by Model\n'
                       +' (Total N=' + str(all_data.shape[0]) + ')', struct_var, 'post-covid_allsubj', working_dir)

    #specify which columns of dataframe to use as covariates
    X_test = all_data[['agedays']]

    #make a matrix of response variables, one for each brain region
    y_test = all_data.loc[:, roi_ids]

    #specify paths
    training_dir = '{}/data/{}/ROI_models/'.format(working_dir, struct_var)
    out_dir = '{}/predict_files/{}/ROI_models/'.format(working_dir, struct_var)
    #  this path is where ROI_models folders are located
    predict_files_dir = '{}/predict_files/{}/ROI_models/'.format(working_dir, struct_var)

    ##########
    # Create output directories for each region and place covariate and response files for that region in  each directory
    ##########
    for c in y_test.columns:
        y_test[c].to_csv(f'{working_dir}/resp_te_'+c+'.txt', header=False, index=False)
        X_test.to_csv(f'{working_dir}/cov_te.txt', sep='\t', header=False, index=False)
        y_test.to_csv(f'{working_dir}/resp_te.txt', sep='\t', header=False, index=False)

    for i in roi_ids:
        roidirname = '{}/predict_files/{}/ROI_models/{}'.format(working_dir, struct_var, i)
        makenewdir(roidirname)
        resp_te_filename = "{}/resp_te_{}.txt".format(working_dir, i)
        resp_te_filepath = roidirname + '/resp_te.txt'
        shutil.copyfile(resp_te_filename, resp_te_filepath)
        cov_te_filepath = roidirname + '/cov_te.txt'
        shutil.copyfile("{}/cov_te.txt".format(working_dir), cov_te_filepath)

    movefiles("{}/resp_*.txt".format(working_dir), "{}/predict_files/{}/response_files/"
              .format(working_dir, struct_var))
    movefiles("{}/cov_t*.txt".format(working_dir), "{}/predict_files/{}/covariate_files/"
              .format(working_dir, struct_var))

    # # Create Design Matrix and add in spline basis and intercept
    # create_design_matrix_one_gender('test', agemin, agemax, spline_order, spline_knots, roi_ids, out_dir)

    # Create dataframe to store Zscores
    Z_time2 = pd.DataFrame()
    Z_time2['participant_id'] = all_data['participant_id'].copy()
    Z_time2.reset_index(inplace=True, drop = True)

    ####Make Predictions of Brain Structural Measures Post-Covid based on Pre-Covid Normative Model

    #create design matrices for all regions and save files in respective directories
    create_design_matrix_one_gender('test', agemin, agemax, spline_order, spline_knots, roi_ids, predict_files_dir)

    for roi in roi_ids:
        print('Running ROI:', roi)
        roi_dir = os.path.join(predict_files_dir, roi)
        model_dir = os.path.join(training_dir, roi, 'Models')
        # os.chdir(roi_dir)

        # configure the covariates to use.
        cov_file_te = os.path.join(roi_dir, 'cov_bspline_te.txt')

        # load test response files
        resp_file_te = os.path.join(roi_dir, 'resp_te.txt')

        # make predictions
        yhat_te, s2_te, Z = predict(cov_file_te, respfile=resp_file_te, alg='blr', model_path=model_dir)

        Z_time2[roi] = Z

        #create dummy design matrices
        dummy_cov_file_path= \
            create_dummy_design_matrix_one_gender(struct_var, agemin, agemax, cov_file_te, spline_order, spline_knots,
                                                  working_dir)

        plot_data_with_spline_one_gender(gender, 'Postcovid (Test) Data ', struct_var, cov_file_te, resp_file_te,
                                         dummy_cov_file_path, model_dir, roi, show_plots, working_dir)

        mystop=1

    Z_time2.to_csv('{}/predict_files/{}/Z_scores_by_region_postcovid_testset_Final.txt'
                                .format(working_dir, struct_var), index=False)

    plt.show()

    return Z_time2

