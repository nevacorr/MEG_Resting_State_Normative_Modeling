# uncompyle6 version 3.9.1
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.9.18 (main, Sep 11 2023, 13:41:44) 
# [GCC 11.2.0]
# Embedded file name: /home/toddr/neva/PycharmProjects/MEG Resting State Normative Modeling/prepare_rsMEG_data.py
# Compiled at: 2024-01-11 11:50:41
# Size of source mod 2**32: 2198 bytes
import pandas as pd

def prepare_rsMEG_data(workingdir, filename):
    resting_state_data = pd.read_csv(workingdir + '/' + filename)
    resting_state_data.rename(columns={"id": "subject"}, inplace=True)
    rsd_v1 = resting_state_data.filter(regex="subject|t1_").copy()
    rsd_v2 = resting_state_data.filter(regex="subject|t2_").copy()
    rsd_v1.dropna(axis=0, thresh=10, inplace=True, ignore_index=True)
    rsd_v2.dropna(axis=0, thresh=10, inplace=True, ignore_index=True)
    rsd_v1.drop((rsd_v1[rsd_v1["subject"] == 532].index), inplace=True)
    MPF_behav_sm = pd.read_csv("/home/toddr/neva/PycharmProjects/AdolescentBehavioral/MPF_Behav_SocialM_v1and2.csv")
    agedays_df = MPF_behav_sm[["subject", "visit", "agedays"]].copy()
    agedays_df.dropna(inplace=True, ignore_index=True)
    agedays_df["agedays"] = agedays_df["agedays"].astype(int)
    rsd_v1.rename(columns={"t1_age": "agegrp"}, inplace=True)
    rsd_v2.rename(columns={"t2_age": "agegrp"}, inplace=True)
    agedays_v1 = agedays_df[agedays_df["visit"] == 1]
    agedays_v1 = agedays_v1.drop(columns=["visit"])
    rsd_v1 = rsd_v1.merge(agedays_v1, how="left", on=["subject"])
    agecol = rsd_v1.pop("agedays")
    rsd_v1.insert(2, "agedays", agecol)
    agedays_v2 = agedays_df[agedays_df["visit"] == 2]
    agedays_v2 = agedays_v2.drop(columns=["visit"])
    rsd_v2 = rsd_v2.merge(agedays_v2, how="left", on=["subject"])
    agecol = rsd_v2.pop("agedays")
    rsd_v2.insert(2, "agedays", agecol)
    gender_v1 = [2 if x % 2 == 0 else 1 for x in rsd_v1.subject]
    rsd_v1.insert(3, "gender", pd.Series(gender_v1))
    gender_v2 = [2 if x % 2 == 0 else 1 for x in rsd_v2.subject]
    rsd_v2.insert(3, "gender", pd.Series(gender_v2))
    return (rsd_v1, rsd_v2)
