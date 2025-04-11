# uncompyle6 version 3.9.1
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.9.18 (main, Sep 11 2023, 13:41:44) 
# [GCC 11.2.0]
# Embedded file name: /home/toddr/neva/PycharmProjects/MEG Resting State Normative Modeling/prepare_rsMEG_data.py
# Compiled at: 2024-01-11 11:50:41
# Size of source mod 2**32: 2198 bytes
import pandas as pd
import sys

def prepare_rsMEG_data(filename, subjects_to_exclude, ct_data_dir):

    # Read in rsMEG data from file
    resting_state_data = pd.read_csv(filename)

    # Rename subject column
    resting_state_data.rename(columns={"id": "subject"}, inplace=True)

    # Remove subjects to exclude
    resting_state_data = resting_state_data[~resting_state_data['subject'].isin(subjects_to_exclude)]

    # Separate v1 and v2 data into separate dataframes
    rsd_v1 = resting_state_data.filter(regex="subject|t1_").copy()
    rsd_v2 = resting_state_data.filter(regex="subject|t2_").copy()

    # Drop rows with more than 10 missing values
    rsd_v1.dropna(axis=0, thresh=10, inplace=True, ignore_index=True)
    rsd_v2.dropna(axis=0, thresh=10, inplace=True, ignore_index=True)

    # Check for nan values in data
    nan_rows1 = rsd_v1.isna().sum().sum()
    nan_rows2 = rsd_v2.isna().sum().sum()
    if (nan_rows1 != 0) and (nan_rows2 != 0):
        sys.exit('Error: Input data has nan values. Stopping program execution.')

    # Rename column in MEG dataframe that has age group
    rsd_v1.rename(columns={"t1_age": "agegrp"}, inplace=True)
    rsd_v2.rename(columns={"t2_age": "agegrp"}, inplace=True)

    # Read in data from cortical thickness data file
    ct_df = pd.read_csv(f"{ct_data_dir}/Adol_CortThick_data.csv")

    # Keep only demographics data
    demographics_df = ct_df[["subject", "gender", "visit", "agedays"]].copy()

    # Remove rows with missing values
    demographics_df.dropna(inplace=True, ignore_index=True)

    # Convert agedays and gender to type int
    demographics_df["agedays"] = demographics_df["agedays"].astype(int)
    demographics_df["gender"] = demographics_df["gender"].astype(int)

    # Extract only data from visit 1 from demographics dataframe
    demo_v1 = demographics_df[demographics_df["visit"] == 1]

    # Remove visit column from the extracted data
    demo_v1 = demo_v1.drop(columns=["visit"])

    # Merge demographic data to MEG dataframe based on subject number
    rsd_v1 = rsd_v1.merge(demo_v1, how="left", on=["subject"])

    # Remove and save agedays column
    agecol = rsd_v1.pop("agedays")

    # Remove and save gender column
    gendercol = rsd_v1.pop("gender")

    # Insert agedays column for visit 1 to second column in dataframe
    rsd_v1.insert(2, "agedays", agecol)

    # Insert gender column for visit 1 to third column in dataframe
    rsd_v1.insert(3, "gender", gendercol)

    # Extract only data from visit 2 from demographics dataframe
    demo_v2 = demographics_df[demographics_df["visit"] == 2]

    # Remove visit column from the extracted data
    demo_v2 = demo_v2.drop(columns=["visit"])

    # Merge demographic data to MEG dataframe based on subject number
    rsd_v2 = rsd_v2.merge(demo_v2, how="left", on=["subject"])

    # Remove and save agedays column
    agecol = rsd_v2.pop("agedays")

    # Remove and save gender column
    gendercol = rsd_v2.pop("gender")

    # Insert agedays column for visit 2 to second column in dataframe
    rsd_v2.insert(2, "agedays", agecol)

    # Insert gender column for visit 2 to thir column in v2 dataframe
    rsd_v2.insert(3, "gender", gendercol)

    # Make lists of subjects with data only at timepoint 1, subjects with data at only timepoint 2, and all subjects in dataset
    # Make a dataframe with visit and subject data from both visits
    # Add a 'visit' column to each DataFrame (without altering the original ones)
    rsd_v1_with_visit = rsd_v1[['subject']].copy()
    rsd_v1_with_visit['visit'] = 1
    rsd_v2_with_visit = rsd_v2[['subject']].copy()
    rsd_v2_with_visit['visit'] = 2
    # Concatenate them
    rsd_allvisits = pd.concat([rsd_v1_with_visit, rsd_v2_with_visit], ignore_index=True)
    all_subjects =rsd_allvisits['subject'].unique().tolist()

    unique_subjects = rsd_allvisits['subject'].value_counts()
    unique_subjects = unique_subjects[unique_subjects == 1].index
    subjects_with_one_dataset = rsd_allvisits[rsd_allvisits['subject'].isin(unique_subjects)]
    subjects_visit1_data_only = subjects_with_one_dataset[subjects_with_one_dataset['visit'] == 1]
    subjects_visit2_data_only = subjects_with_one_dataset[subjects_with_one_dataset['visit'] == 2]
    subjects_v1_only = subjects_visit1_data_only['subject'].tolist()
    subjects_v2_only = subjects_visit2_data_only['subject'].tolist()

    return (rsd_v1, rsd_v2, all_subjects, subjects_v1_only, subjects_v2_only)
