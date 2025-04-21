import os
import pandas as pd
import matplotlib.pyplot as plt
import myggseg

from helper_functions_MEG import read_text_list

# Set some options
lobes_only = 0

working_dir = os.getcwd()

frontal_reg = ['superiorfrontal', 'rostralmiddlefrontal', 'caudalmiddlefrontal', 'parsopercularis', 'parstriangularis',
               'parsorbitalis', 'lateralorbitofrontal', 'medialorbitofrontal', 'precentral', 'paracentral',
               'frontalpole',
               'rostralanteriorcingulate', 'caudalanteriorcingulate']

parietal_reg = ['superiorparietal', 'inferiorparietal', 'supramarginal', 'postcentral', 'precuneus',
                'posteriorcingulate',
                'isthmuscingulate']

temporal_reg = ['superiortemporal', 'middletemporal', 'inferiortemporal', 'bankssts', 'fusiform', 'transversetemporal',
                'entorhinal', 'temporalpole', 'parahippocampal']

occipital_reg = ['lateraloccipital', 'lingual', 'cuneus', 'pericalcarine']

regions_reject_female_file = f'{working_dir}/output_data/regions_reject_null_rsMEG_female.csv'
regions_female = read_text_list(regions_reject_female_file)

regions_reject_male_file = f'{working_dir}/output_data/regions_reject_null_rsMEG_male.csv'
regions_male = read_text_list(regions_reject_male_file)

rfemale = {}
rmale = {}
female_dict = {}
male_dict = {}

for i, band in enumerate(['theta','alpha', 'beta', 'gamma']):
    rfemale[band] = [item for item in regions_female if band in item]
    rmale[band] = [item for item in regions_male if band in item]

    female_dict[band] = {}
    male_dict[band] = {}

    for reg in rfemale[band]:
        region = reg.replace(f'_{band}', '')
        if '-lh' in region:
            region = region.replace('-lh', '_left')
        else:
            region = region.replace('-rh', '_right')
        female_dict[band][region] = i

    for reg in rmale[band]:
        region = reg.replace(f'_{band}', '')
        if '-lh' in region:
            region = region.replace('-lh', '_left')
        else:
            region = region.replace('-rh', '_right')
        male_dict[band][region] = i

    save_dir = working_dir + '/plots'

    female_dict_to_plot = female_dict[band].copy()
    male_dict_to_plot = female_dict[band].copy()

    filename = f'Female Regions with significantly altered power in post-COVID rsMEG {band} band'
    myggseg.plot_dk(female_dict_to_plot, save_dir, filename, cmap='Dark2', background='k', vminmax=[0, 3], edgecolor='w', bordercolor='gray', figsize=(8,8),
                      title=f'Female Regions with Significantly Reduced Power in\nPost-COVID rsMEG {band.capitalize()} Band Power')

    filename = f'Male Regions with significantly altered power in post-COVID rsMEG {band} band'
    myggseg.plot_dk(male_dict_to_plot, save_dir, filename, cmap='Dark2', background='k', vminmax=[0, 3], edgecolor='w', bordercolor='gray', figsize=(8,8),
                      title=f'Male Regions with Significantly Reduced Power in\nPost-COVID rsMEG {band.capitalize()} Band Power')

plt.show()
mystop=1