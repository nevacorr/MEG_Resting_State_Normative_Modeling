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

    return (rsd_v1, rsd_v2)
