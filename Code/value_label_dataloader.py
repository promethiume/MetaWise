import sys

import pandas as pd
import numpy as np
import yaml
import os

def loader():

    if os.path.exists("value_label_settings.yml"):
        settings = yaml.safe_load(open("value_label_settings.yml", "r"))
    else:
        print("Expected a file settings.yml but didn't find it.")
        return

    # get input
    print('--> Acquiring data...')
    value_file = settings['data']['value_file']
    label_file = settings['data']['label_file']
    value = pd.read_csv(value_file, sep="\t")
    labels = pd.read_csv(label_file, sep="\t")
    print('Finished acquiring data.')

    column2 = settings['data']['type_description']['type_description_column']
    column2_1 = settings['data']['type_description']['type_description_Primary']
    column2_2 = settings['data']['type_description']['type_description_Metastatic']

    # define labels
    primary_labels = labels[labels[column2] == column2_1]
    metastatic_labels = labels[labels[column2] == column2_2]
    print(len(labels))
    print(len(primary_labels))
    print(len(metastatic_labels))
    column3 = settings['data']['Tumor_Sample_Barcode']
    # print(primary_labels[column3])
    primary = primary_labels[column3].values.tolist()
    metastatic = metastatic_labels[column3].values.tolist()

    # remove data without labels
    primary.remove('TCGA-06-0165-01')
    primary.remove('TCGA-G9-6370-01')
    primary.remove('TCGA-YU-A90S-01')
    primary.remove('TCGA-AB-2837-03')
    primary.remove('TCGA-AB-2823-03')
    primary.remove('TCGA-AB-2840-03')
    metastatic.remove('TP_2090')

    # define data
    primary_value = value[primary].values.tolist()
    metastatic_value = value[metastatic].values.tolist()

    primary_values = np.array(primary_value)
    metastatic_values = np.array(metastatic_value)

    if settings['data']['flags'] == 'column':
        primary_values = primary_values.T
        metastatic_values = metastatic_values.T
    print(len(primary_values))
    print(len(metastatic_values))

    primary_num = int(settings['data']['primary_num'])
    metastatic_num = int(settings['data']['metastatic_num'])

    # shuffle, random select data
    row_rand_primary_values = np.arange(primary_values.shape[0])
    np.random.shuffle(row_rand_primary_values)
    # primary_values = primary_values[row_rand_primary_values[0:primary_num]]
    row_rand_metastatic_values = np.arange(metastatic_values.shape[0])
    np.random.shuffle(row_rand_metastatic_values)
    metastatic_values = metastatic_values[row_rand_metastatic_values[0:metastatic_num]]
    datas = []
    K = int(settings['data']['K'])

    # concate data and labels
    for i in range(K):
        start = i * primary_num
        end = start + primary_num
        datas.append(np.concatenate((primary_values[row_rand_primary_values[start:end]], metastatic_values), axis=0))
    labels = np.concatenate((np.zeros(primary_num), np.ones(metastatic_num)), axis=0)
    return datas, labels
