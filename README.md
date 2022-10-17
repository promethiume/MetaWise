# MetaWise

Deep Learning model of metastatic tumors classifier

https://www.biorxiv.org/content/10.1101/2022.09.29.510207v1

## Code

├── category_dataloader.py              `load category type data`

├── category_settings.yml               `load category type data`

├── train_models.py                     `model training code`

├── Mutational_Signatures_Analysis_for_WES_data.R     `data pre-processing code`

├── value_label_dataloader.py           `load value/label data`

└── value_label_settings.yml            `load value/label data`

## Getting Started
1. pre-requirements:

    tensorflow-gpu 2.4.0
    
    python 3.7.13
    
    pandas 1.3.5
   
    numpy 1.19.5
    
    keras 2.9.0
    
    cudnn 8.0.5.39
    
    cuda 11.0.3
    
2. model training:

python3 train_models.py
