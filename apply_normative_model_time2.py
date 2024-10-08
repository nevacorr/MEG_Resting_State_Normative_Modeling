#####
# This program imports the model and Z-scores from the bayesian linear regression normative modeling of the
# training data set (which is the adolescent visit 1 data). It then uses the model to calculate Z-scores for
# the post-covid adolescent (visit 2) data.
# Author: Neva M. Corrigan
######
import os
import pandas as pd
from matplotlib import pyplot as plt
from prepare_rsMEG_data import prepare_rsMEG_data
from helper_functions_MEG import plot_num_subjs
from helper_functions_MEG import makenewdir, movefiles, create_design_matrix_one_gender
from helper_functions_MEG import plot_data_with_spline_one_gender_rescale, create_dummy_design_matrix_one_gender, read_ages_from_file
import shutil
from normative_edited import predict
from joblib import load

def apply_normative_model_time2(gender, struct_var, show_plots, show_nsubject_plots, spline_order, spline_knots,
                                working_dir, MEG_filename, ct_data_dir, subjects_to_exclude, bands, lobes_only):

    # load all rs MEG data
    rsd_v1, rsd_v2 = prepare_rsMEG_data(MEG_filename, subjects_to_exclude, ct_data_dir)

    if gender == 'male':
        # keep only data for males
        rsd_v2 = rsd_v2.loc[rsd_v2['gender'] == 1]
    else:
        # keep only data for females
        rsd_v2 = rsd_v2.loc[rsd_v2['gender'] == 2]

    #remove sex column
    rsd_v2 = rsd_v2.drop(columns=['gender'])

    # Remove the prefix 't2_' from column names
    rsd_v2.columns = rsd_v2.columns.str.replace(r'^t2_', '', regex=True)

    if lobes_only:
        #Average values for all regions within each lobe
        frontal_reg = ['superiorfrontal', 'rostralmiddlefrontal', 'caudalmiddlefrontal', 'parsopercularis',
                       'parstriangularis',
                       'parsorbitalis', 'lateralorbitofrontal', 'medialorbitofrontal', 'precentral', 'paracentral',
                       'frontalpole',
                       'rostralanteriorcingulate', 'caudalanteriorcingulate']

        parietal_reg = ['superiorparietal', 'inferiorparietal', 'supramarginal', 'postcentral', 'precuneus',
                        'posteriorcingulate',
                        'isthmuscingulate']

        temporal_reg = ['superiortemporal', 'middletemporal', 'inferiortemporal', 'bankssts', 'fusiform',
                        'transversetemporal',
                        'entorhinal', 'temporalpole', 'parahippocampal']

        occipital_reg = ['lateraloccipital', 'lingual', 'cuneus', 'pericalcarine']

        region_dict = {
            'frontal': frontal_reg,
            'parietal': parietal_reg,
            'temporal': temporal_reg,
            'occipital': occipital_reg
        }

        results_df = pd.DataFrame(index=rsd_v2.index)

        hemispheres = ['-lh', '-rh']

        for band in bands:
            for region_name, regions in region_dict.items():
                for hemi in hemispheres:
                    # Create a pattern to match the columns of interest
                    cols_to_avg = [col for col in rsd_v2.columns if
                                   any(f'{band}_{region}{hemi}' in col for region in regions)]

                    if cols_to_avg:
                        # Average the values across columns in the region
                        results_df[f'{band}_{region_name}{hemi}'] = rsd_v2[cols_to_avg].mean(axis=1)

        # Merge the new averaged columns with the original dataframe
        # This will overwrite the matching columns but keep the other columns unchanged
        rsd_v2 = rsd_v2.drop(columns=[col for col in rsd_v2.columns if
                              any(band in col for band in bands)])  # Remove original band-region columns
        rsd_v2 = pd.concat([rsd_v2, results_df], axis=1)


    # Scale non-categorical covariate and response variables using same scaling a time 1 data
    cols_to_eval = [col for col in rsd_v2.columns if '-lh' in col or '-rh' in col]
    cols_to_eval.append('agedays')
    minmax_scaler = load(f'{working_dir}/minmax_scaler_{gender}.bin')
    rsd_v2[cols_to_eval] = minmax_scaler.transform(rsd_v2[cols_to_eval])

    ########
    # Use same train test subgroups as was used for cortical thickness analysis
    ########

    #extract subject numbers from visit 1 and find subjects in visit 2 that aren't in visit 1
    rows_in_v2_but_not_v1 = rsd_v2[~rsd_v2['subject'].isin(rsd_v1['subject'])].dropna()
    subjs_in_v2_not_v1 = rows_in_v2_but_not_v1['subject'].copy()
    subjs_in_v2_not_v1 = subjs_in_v2_not_v1.astype(int)

    #only keep subjects at 12, 14 and 16 years of age (subject numbers <400) because cannot model 18 and 20 year olds
    subjs_in_v2_not_v1 = subjs_in_v2_not_v1[subjs_in_v2_not_v1 < 400]

    #only include subjects that were not in the training set
    fname='{}/visit1_subjects_excluded_from_normative_model_test_set_{}_9_11_13.txt'.format(ct_data_dir, 'cortthick')
    subjects_to_include = pd.read_csv(fname, header=None)
    subjects_to_include = pd.concat([subjects_to_include, subjs_in_v2_not_v1])
    rsd_v2 = rsd_v2[rsd_v2['subject'].isin(subjects_to_include[0])]

    # reset indices
    rsd_v2.reset_index(inplace=True)

    #show number of subjects by gender and age
    if gender == "female":
        genstring = 'Female'
    elif gender == "male":
        genstring = 'Male'

    if show_nsubject_plots:
        plot_num_subjs(gender, rsd_v2, f'{genstring} Subjects by Age with Post-COVID MEGrs Data\nEvaluated by Model\n'
                       +' (Total N=' + str(rsd_v2.shape[0]) + ')', struct_var, 'post-covid_allsubj', working_dir)

    # Remove the prefix 't2_' from column names
    rsd_v2.columns = rsd_v2.columns.str.replace(r'^t2_', '', regex=True)

    # read agemin and agemax from file
    agemin, agemax = read_ages_from_file(struct_var, working_dir, gender)

    #specify which columns of dataframe to use as covariates
    rs_covariates = rsd_v2[['agegrp', 'agedays']]

    #make a matrix of response variables, one for each brain region
    rscols = [col for col in rsd_v2.columns if col not in ['subject', 'agegrp', 'agedays']]

    # make file diretories for output
    makenewdir('{}/predict_files/'.format(working_dir))

    # loop through each power band separately
    for bandnum, band in enumerate(bands):

        # make file diretories for band-specific output

        makenewdir('{}/predict_files/{}_{}'.format(working_dir, gender, band))
        makenewdir('{}/predict_files/{}_{}/plots'.format(working_dir, gender, band))
        makenewdir('{}/predict_files/{}_{}/ROI_models'.format(working_dir, gender, band))
        makenewdir('{}/predict_files/{}_{}/covariate_files'.format(working_dir, gender, band))
        makenewdir('{}/predict_files/{}_{}/response_files'.format(working_dir, gender, band))

        rscols_band = [item for item in rscols if band in item]

        rs_features = rsd_v2.loc[:, rscols_band]

        X_test = rs_covariates.copy()
        y_test = rs_features.copy()

        # for the first loop iteration, save the subject numbers for the training and validation sets to variables
        if band == 'theta':
            s_index_test = X_test.index.values
            subjects_test = rsd_v2.loc[s_index_test, 'subject'].values

        # drop the agegrp column from the test data set because we want to use agedays as a predictor
        X_test.drop(columns=['agegrp'], inplace=True)

        # Get region names from test set feature names
        roi_ids = y_test.columns.str.replace(fr'^{band}_', '', regex=True).tolist()

        ##########
        # Create output directories for each region and place covariate and response files for that region in  each directory
        ##########
        for c in y_test.columns:
            y_test[c].to_csv(f'{working_dir}/resp_te_' + gender + '_' + c + '.txt', header=False, index=False)
            X_test.to_csv(f'{working_dir}/cov_te.txt', sep='\t', header=False, index=False)
            y_test.to_csv(f'{working_dir}/resp_te.txt', sep='\t', header=False, index=False)

        for i in roi_ids:
            roidirname = '{}/predict_files/{}_{}/ROI_models/{}'.format(working_dir, gender, band, i)
            makenewdir(roidirname)
            resp_te_filename = "{}/resp_te_{}_{}_{}.txt".format(working_dir, gender, band, i)
            resp_te_filepath = roidirname + '/resp_te.txt'
            shutil.copyfile(resp_te_filename, resp_te_filepath)
            cov_te_filepath = roidirname + '/cov_te.txt'
            shutil.copyfile("{}/cov_te.txt".format(working_dir), cov_te_filepath)

        movefiles("{}/resp_*.txt".format(working_dir), "{}/predict_files/{}_{}/response_files/"
                  .format(working_dir,gender, band))
        movefiles("{}/cov_t*.txt".format(working_dir), "{}/predict_files/{}_{}/covariate_files/"
                  .format(working_dir, gender, band))

        # specify paths
        training_dir = '{}/data/{}_{}/ROI_models/'.format(working_dir, gender, band)
        out_dir = '{}/predict_files/{}_{}/ROI_models/'.format(working_dir, gender, band)
        #  this path is where ROI_models folders are located
        predict_files_dir = '{}/predict_files/{}_{}/ROI_models/'.format(working_dir, gender, band)

        # Create Design Matrix and add in spline basis and intercept
        create_design_matrix_one_gender('test', agemin, agemax, spline_order, spline_knots, roi_ids, out_dir)

        # Create dataframe to store Zscores
        subjects_test = subjects_test.reshape(-1, 1)
        Z_score_test_matrix = pd.DataFrame(subjects_test, columns=['subject_id_test'])

        ####Make Predictions of Brain Structural Measures Post-Covid based on Pre-Covid Normative Model

        for regnum, roi in enumerate(roi_ids):
            print('Running ROI:', roi)
            roi_dir = os.path.join(predict_files_dir, roi)
            model_dir = os.path.join(training_dir, roi, 'Models')
            os.chdir(roi_dir)

            # configure the covariates to use.
            cov_file_te = os.path.join(roi_dir, 'cov_bspline_te.txt')

            # load test response files
            resp_file_te = os.path.join(roi_dir, 'resp_te.txt')

            # make predictions
            yhat_te, s2_te, Z = predict(cov_file_te, respfile=resp_file_te, alg='blr', model_path=model_dir)

            #create dummy design matrices
            dummy_cov_file_path= create_dummy_design_matrix_one_gender(band, agemin, agemax,
                                                        cov_file_te, spline_order, spline_knots, working_dir)

            totalregnum = len(roi_ids)

            plot_data_with_spline_one_gender_rescale(gender, 'Postcovid (Test) Data ', band, cov_file_te, resp_file_te,
                        dummy_cov_file_path, model_dir, roi, show_plots, working_dir, minmax_scaler, regnum, bandnum, totalregnum)

            Z_score_test_matrix[roi] = Z


            Z_score_test_matrix.to_csv('{}/predict_files/{}_{}/Z_scores_by_region_postcovid_testset_Final.txt'
                                .format(working_dir, gender, band), index=False)

            plt.show()

    return Z_score_test_matrix, roi_ids

