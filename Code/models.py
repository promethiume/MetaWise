import time

import tensorflow as tf
import random as rn
import numpy as np
from tensorflow.keras import backend as K
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
# Setting the seed for numpy-generated random numbers
np.random.seed(45)

# Setting the graph-level random seed.
tf.random.set_seed(1337)

rn.seed(73)

from tensorflow.compat.v1.keras.backend import set_session

session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=0,
    inter_op_parallelism_threads=0)

import math
import pandas as pd

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

import sys


if __name__ == '__main__':
    print("请输入训练集占比(小数)：")
    fol = float(0.8)
    if fol <= 0 or fol >= 1:
        sys.exit()
    fold = int(fol * 100)
    test_size = 1 - fol
    print(test_size)
    # feature_type = sys.argv[2]

    print("输入文件为按类别分布，请输入0；输入文件按data和value分布，请输入1：")
    tp = int(1)
    if tp == 0:
        from category_dataloader import loader
    elif tp == 1:
        from value_label_dataloader import loader
    datas, labels = loader()
    for i in range(len(datas)):
        path_best_model = './best_model-DNN-data7-107_%s.keras' % fold
        x_train, x_val_test, y_train, y_val_test = train_test_split(datas[i], labels, test_size=test_size,
                                                                    random_state=42)
        x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=42)

        encoder = LabelEncoder()
        # test_labels_names = y_test[i]
        y_test = encoder.fit_transform(y_test)
        test_labels = y_test
        y_test = tensorflow.keras.utils.to_categorical(y_test, 2)
        y_train = encoder.fit_transform(y_train)
        train_labels = y_train
        y_train = tensorflow.keras.utils.to_categorical(y_train, 2)
        y_val = encoder.fit_transform(y_val)
        val_labels = y_val
        y_val = tensorflow.keras.utils.to_categorical(y_val, 2)
        input_size = x_train.shape[1]
        num_classes = 2

        model = load_model(path_best_model)

        callback_log = tf.compat.v1.keras.callbacks.TensorBoard(
            histogram_freq=0,
            batch_size=64,
            write_graph=True,
            write_grads=True,
            write_images=False)
        callbacks = [callback_log]
        validation_data = (x_val, y_val)

        model.fit(x=x_train, y=y_train, epochs=100, batch_size=64, validation_data=validation_data,
                                callbacks=callbacks)

        # Evaluate best model on test data
        y_train_pre = model.predict(x_train)
        y_train_pred = np.argmax(y_train_pre, axis=1)
        y_val_pre = model.predict(x_val)
        y_val_pred = np.argmax(y_val_pre, axis=1)
        y_test_pre = model.predict(x_test)
        y_test_pred = np.argmax(y_test_pre, axis=1)
        train_report = classification_report(train_labels, y_train_pred)
        val_report = classification_report(val_labels, y_val_pred)
        test_report = classification_report(test_labels, y_test_pred)
        # Save best model
        model.save('./model/DNN-data7-107_%s_%s.keras' % (fold, i))
        from tensorflow.keras.utils import model_to_dot

        src = model_to_dot(model, show_shapes=True,
                           show_layer_names=True,
                           rankdir='TB',
                           dpi=72, expand_nested=True, subgraph=False)
        src.write_svg("./model/model_summary-data7-107_%s.svg" % fold)
        with open("./results/DNN_results-data7_107_%s.txt" % fold, "a") as f:
            f.write('model_name: DNN-data7-107_%s_%s.keras' % (fold, i) + '\n')
            f.write("训练集：" + '\n')
            f.write(str(train_report))
            f.write("验证集：" + '\n')
            f.write(str(val_report))
            f.write("测试集：" + '\n')
            f.write(str(test_report))
            f.write('\n')
