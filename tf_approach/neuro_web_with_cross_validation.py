from typing import Any

from read_data.read_xls import read_xls_ziwm
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
from pandas import DataFrame
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


def split_data_and_lables(data: DataFrame, divide_value: int = 8):
    data_list = []
    lable_list = []
    for index, row in data.iterrows():
        lable_list.append(row['Unnamed: 0'])
        data_list.append([i / divide_value for i in row[2:22]])
    return np.array(data_list), np.array([lable - 1 for lable in lable_list])


def k_time_cross_validation(model: Any, name_model: str, data: DataFrame, k_time: int = 5):
    results = []
    for i in range(k_time):
        train_raw_data, test_raw_data = train_test_split(data, test_size=0.5)
        train_data, train_lables = split_data_and_lables(train_raw_data)
        test_data, test_lables = split_data_and_lables(test_raw_data)
        model.fit(train_data, train_lables, epochs=200)
        test_loss, test_acc = model.evaluate(test_data, test_lables)
        results.append((test_loss, test_acc))
        logging.info(f'{i} - time')

    loss_sum = 0.0
    acc_sum = 0.0
    for r in results:
        loss_sum += r[0]
        acc_sum += r[1]

    loss_avg = loss_sum / len(results)
    acc_avg = acc_sum / len(results)
    logging.info(f'{name_model} results: ')
    logging.info(f'  loss: {loss_avg}')
    logging.info(f'  acc: {acc_avg}')


if __name__ == '__main__':
    raw_data = read_xls_ziwm('/home/bomba/PycharmProjects/ZIwM/data/bialaczka.XLS')

    _model = keras.Sequential([
        keras.layers.Dense(20, activation=tf.nn.relu),
        keras.layers.Dense(130, activation=tf.nn.relu),
        keras.layers.Dense(20, activation=tf.nn.softmax)
    ])

    _model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    k_time_cross_validation(_model, 'model1', raw_data)


