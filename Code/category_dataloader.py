import sys

import pandas as pd
import numpy as np
import yaml
import os

def loader():

    if os.path.exists("category_settings.yml"):
        settings = yaml.safe_load(open("category_settings.yml", "r"))
    else:
        print("Expected a file settings.yml but didn't find it.")
        return

    # 读取实验数据
    print('--> Acquiring data...')
    primary_file = settings['data']['primary_file']
    primary_value = pd.read_csv(primary_file, sep="\t")

    primary_value = np.array(primary_value)
    primary_values = primary_value.T[0:2000, :]    #1900

    metastatic_file = settings['data']['metastatic_file']
    metastatic_value = pd.read_csv(metastatic_file, sep="\t")

    metastatic_value = np.array(metastatic_value)
    metastatic_values = metastatic_value.T
    print('Finished acquiring data.')

    # 将阴性和阳性数据按行进行连接
    datas = np.concatenate((primary_values, metastatic_values), axis=0)

    primary_num = int(settings['data']['primary_num'])
    # 阴阳性数据类别标签，0为阴性，1为阳性
    labels = []
    for i in range(len(datas)):
        if i < primary_num:
            labels.append('0')
        else:
            labels.append('1')
    labels = np.array(labels)
    # print(datas)
    # sys.exit()
    return datas, labels
