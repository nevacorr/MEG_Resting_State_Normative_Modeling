import os
import pandas as pd
import matplotlib.pyplot as plt

from helper_functions_MEG import read_text_list

working_dir = os.getcwd()

regions_reject_female_file = f'{working_dir}/regions_reject_null_rsMEG_female.csv'
regions_female = read_text_list(regions_reject_female_file)

regions_reject_male_file = f'{working_dir}/regions_reject_null_rsMEG_male.csv'
regions_male = read_text_list(regions_reject_male_file)

mystop=1