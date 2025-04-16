import pandas as pd

def combine_rgions_by_lobe(rsd_v1, rsd_v2, bands):
        # average values for all regions in each lobe

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

    region_dict={
        'frontal': frontal_reg,
        'parietal': parietal_reg,
        'temporal': temporal_reg,
        'occipital': occipital_reg
    }

    results_df = pd.DataFrame(index=rsd_v1.index)

    hemispheres = ['-lh', '-rh']

    for band in bands:
        for region_name, regions in region_dict.items():
            for hemi in hemispheres:
                # Create a pattern to match the columns of interest
                cols_to_avg = [col for col in rsd_v1.columns if any(f'{band}_{region}{hemi}' in col for region in regions)]

                if cols_to_avg:
                    # Average the values across columns in the region
                    results_df[f'{band}_{region_name}{hemi}'] = rsd_v1[cols_to_avg].mean(axis=1)

    # Merge the new averaged columns with the original dataframe
    # This will overwrite the matching columns but keep the other columns unchanged
    rsd_v1 = rsd_v1.drop(columns=[col for col in rsd_v1.columns if
                          any(band in col for band in bands)])  # Remove original band-region columns
    rsd_v1 = pd.concat([rsd_v1, results_df], axis=1)

    return rsd_v1, rsd_v2