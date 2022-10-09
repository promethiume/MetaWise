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

    # 读取实验数据
    print('--> Acquiring data...')
    value_file = settings['data']['value_file']
    label_file = settings['data']['label_file']
    value = pd.read_csv(value_file, sep="\t")
    labels = pd.read_csv(label_file, sep="\t")
    print('Finished acquiring data.')

    # column1 = settings['data']['category']['category_column']
    # 统计读入的实验数据包含的癌症类别，可有可无
    # tumor_types = []
    # for i in label[column1].values.tolist():
    #     if i not in tumor_types:
    #         tumor_types.append(i)
    # print(tumor_types)
    # column1_1 = settings['data']['category']['category_name']
    # 选取自己想要的癌症类别
    # labels = label[label[column1] == column1_1]

    column2 = settings['data']['type_description']['type_description_column']
    column2_1 = settings['data']['type_description']['type_description_Primary']
    column2_2 = settings['data']['type_description']['type_description_Metastatic']
    # 从选取的癌症类别中选取阴阳性标签
    primary_labels = labels[labels[column2] == column2_1]
    metastatic_labels = labels[labels[column2] == column2_2]
    print(len(labels))
    print(len(primary_labels))
    print(len(metastatic_labels))
    column3 = settings['data']['Tumor_Sample_Barcode']
    # print(primary_labels[column3])
    primary = primary_labels[column3].values.tolist()
    metastatic = metastatic_labels[column3].values.tolist()
    primary.remove('TCGA-06-0165-01')
    primary.remove('TCGA-G9-6370-01')
    primary.remove('TCGA-YU-A90S-01')
    primary.remove('TCGA-AB-2837-03')
    primary.remove('TCGA-AB-2823-03')
    primary.remove('TCGA-AB-2840-03')
    metastatic.remove('TP_2090')
    # 选取当前癌症类别中阴阳性特征值
    primary_value = value[primary].values.tolist()
    metastatic_value = value[metastatic].values.tolist()

    # 将提取到的list型特征数据转换为numpy类型
    primary_values = np.array(primary_value)
    metastatic_values = np.array(metastatic_value)

    if settings['data']['flags'] == 'column':
        # 将选区的特征数据行列转换，行为癌变情况（阴性、阳性），列为特征值
        primary_values = primary_values.T
        metastatic_values = metastatic_values.T
    print(len(primary_values))
    print(len(metastatic_values))

    primary_num = int(settings['data']['primary_num'])
    metastatic_num = int(settings['data']['metastatic_num'])
    # 按行打乱数据，随机选取阴性阳性数据各一百条
    row_rand_primary_values = np.arange(primary_values.shape[0])
    np.random.shuffle(row_rand_primary_values)
    # primary_values = primary_values[row_rand_primary_values[0:primary_num]]
    row_rand_metastatic_values = np.arange(metastatic_values.shape[0])
    np.random.shuffle(row_rand_metastatic_values)
    metastatic_values = metastatic_values[row_rand_metastatic_values[0:metastatic_num]]
    datas = []
    K = int(settings['data']['K'])
    # 将阴性和阳性数据按行进行连接，标签也进行连接操作
    for i in range(K):
        start = i * primary_num
        end = start + primary_num
        datas.append(np.concatenate((primary_values[row_rand_primary_values[start:end]], metastatic_values), axis=0))
    labels = np.concatenate((np.zeros(primary_num), np.ones(metastatic_num)), axis=0)
    return datas, labels
