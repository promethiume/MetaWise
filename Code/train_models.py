import time

import tensorflow as tf
import random as rn
import numpy as np
from tensorflow.keras import backend as K
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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

dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform', name='learning_rate')

dim_weight_decay = Real(low=1e-5, high=0.5, prior='log-uniform', name='weight_decay')

dim_num_dense_layers = Integer(low=2, high=10, name='num_dense_layers')

dim_num_dense_nodes = Integer(low=0, high=1024, name='num_dense_nodes')

dim_activation = Categorical(categories=['relu', 'softplus'], name='activation')

dim_dropout = Real(low=1e-6, high=0.5, prior='log-uniform', name='dropout')

dimensions = [dim_learning_rate, dim_weight_decay, dim_dropout, dim_num_dense_layers, dim_num_dense_nodes,
              dim_activation]

default_paramaters = [1e-4, 1e-3, 1e-6, 3, 30, 'relu']


def log_dir_name(learning_rate, weight_decay, num_dense_layers, num_dense_nodes, activation):
    s = "./logs/data7-107_%s/lr_{0:.0e}_wd_{0:.0e}_layers_{2}_nodes{3}_{4}/" % fold
    log_dir = s.format(learning_rate, weight_decay, num_dense_layers, num_dense_nodes, activation)
    return log_dir


### Make train test and validaiton here


def create_model(learning_rate, weight_decay, dropout, num_dense_layers, num_dense_nodes, activation):
    ###Define model here
    model = Sequential()
    model.add(InputLayer(input_shape=(input_size,)))
    for i in range(num_dense_layers):
        name = 'layer_dense_{0}'.format(i + 1)
        model.add(
            Dense(num_dense_nodes, activation=activation, name=name, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))
    # optimizer = Adam(lr=learning_rate)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
    return model


@use_named_args(dimensions=dimensions)
def fitness(learning_rate, weight_decay, dropout, num_dense_layers, num_dense_nodes, activation):
    global best_accuracy
    # best_accuracy = 0.0
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model = create_model(learning_rate=learning_rate, weight_decay=weight_decay, dropout=dropout,
                             num_dense_layers=num_dense_layers, num_dense_nodes=num_dense_nodes, activation=activation)
    # print(model)
    # time.sleep(5)
    log_dir = log_dir_name(learning_rate, weight_decay, num_dense_layers,
                           num_dense_nodes, activation)
    callback_log = tf.compat.v1.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        batch_size=32,
        write_graph=True,
        write_grads=True,
        write_images=False)
    callbacks = [callback_log]
    validation_data = (x_val, y_val)

    for i in range(100):
        history = model.fit(x=x_train, y=y_train, epochs=1, batch_size=32, validation_data=validation_data,
                            callbacks=callbacks)
        if i == 0:
            model.save('./DNN-data7-107_%s.keras' % fold)
    accuracy = history.history['val_accuracy'][-1]
    print('Accuracy: {0:.2%}'.format(accuracy))
    if accuracy > best_accuracy:
        model = load_model('./DNN-data7-107_%s.keras' % fold)
        model.save(path_best_model)
        best_accuracy = accuracy
        with open("./model/DNN-data7_107_%s.txt" % fold, "w") as f:
            f.write('model_name: DNN-data7-107_%s.keras' % fold + '\n')
            f.write('learning rate: ' + str(learning_rate) + '\n')
            f.write('weight_decay: ' + str(weight_decay) + '\n')
            f.write('dropout: ' + str(dropout) + '\n')
            f.write('num_dense_layers: ' + str(num_dense_layers) + '\n')
            f.write('num_dense_nodes: ' + str(num_dense_nodes) + '\n')
            f.write('activation: ' + str(activation) + '\n')
    del model
    K.clear_session()
    return -accuracy


if __name__ == '__main__':
    #print("input the train dataset percentage(e.g., 0.8):")
    fol = float(0.8)
    if fol <= 0 or fol >= 1:
        sys.exit()
    fold = int(fol * 100)
    test_size = 1 - fol
    print(test_size)
    # feature_type = sys.argv[2]

    tp = int(1)   # data format. 0 represent "positive/negative data in seperate file"; 1 represent "label in seperate file"
    if tp == 0:
        from category_dataloader import loader
    elif tp == 1:
        from value_label_dataloader import loader
    datas, labels = loader()
    for i in range(1):
        path_best_model = './best_model-DNN-data7-107_%s.keras' % fold
        best_accuracy = 0.0
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

        ### Run Bayesian optimization
        search_result = gp_minimize(func=fitness, dimensions=dimensions, acq_func='EI', n_calls=15,
                                    x0=default_paramaters,
                                    random_state=7, n_jobs=-1)
        # model = load_model(path_best_model)
        # # Evaluate best model on test data
        # y_train_pre = model.predict(x_train)
        # y_train_pred = np.argmax(y_train_pre, axis=1)
        # y_val_pre = model.predict(x_val)
        # y_val_pred = np.argmax(y_val_pre, axis=1)
        # y_test_pre = model.predict(x_test)
        # y_test_pred = np.argmax(y_test_pre, axis=1)
        # train_report = classification_report(train_labels, y_train_pred)
        # val_report = classification_report(val_labels, y_val_pred)
        # test_report = classification_report(test_labels, y_test_pred)
        # # Save best model
        # model.save('./model/DNN-data8-107_%s.keras' % fold)
        # from tensorflow.keras.utils import model_to_dot
        #
        # src = model_to_dot(model, show_shapes=True,
        #                    show_layer_names=True,
        #                    rankdir='TB',
        #                    dpi=72, expand_nested=True, subgraph=False)
        # src.write_svg("./model/model_summary-data8-107_%s_%s.svg" % (fold, i))
        # with open("./results/DNN_results-data8_107_%s.txt" % fold, "a") as f:
        #     f.write('model_name: DNN-data8-107_%s_%s.keras' % (fold, i) + '\n')
        #     f.write("training: " + '\n')
        #     f.write(str(train_report))
        #     f.write("validation: " + '\n')
        #     f.write(str(val_report))
        #     f.write("test: " + '\n')
        #     f.write(str(test_report))
        #     f.write('\n')
