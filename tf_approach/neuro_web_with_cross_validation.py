from typing import Any, Tuple, List

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
        data_list.append([i / divide_value for i in row[[5, 19, 6, 18, 3, 15, 8, 16, 10, 9]]])
    return np.array(data_list), np.array([lable - 1 for lable in lable_list])


# 5, 19, 6, 18, 3, 15, 8, 16, 10, 9

def k_time_cross_validation(model: Any, name_model: str, data: DataFrame, k_time: int = 5) -> Tuple[str, float, float]:
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
    return name_model, loss_avg, acc_avg


# def show_results(results: List[Tuple]):
#     names = [i[0] for i in results]
#     loss = [i[1] for i in results]
#     acc = [i[2] for i in results]
#
#     plt.bar(acc, height=[1,2,3])
#     plt.xticks(acc, names)
#     plt.show()


if __name__ == '__main__':
    INPUT_SIZE = 10
    results_list = []
    raw_data = read_xls_ziwm('/home/bomba/PycharmProjects/ZIwM/data/bialaczka.XLS')

    model_1 = keras.Sequential([
        keras.layers.Dense(INPUT_SIZE, activation=tf.nn.relu),
        keras.layers.Dense(130, activation=tf.nn.relu),
        keras.layers.Dense(20, activation=tf.nn.softmax)
    ])

    model_1.compile(optimizer='rmsprop',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    results_list.append(k_time_cross_validation(model_1, 'model1', raw_data))

    model_2 = keras.Sequential([
        keras.layers.Dense(INPUT_SIZE, activation=tf.nn.relu),
        keras.layers.Dense(200, activation=tf.nn.relu),
        keras.layers.Dense(20, activation=tf.nn.softmax)
    ])

    model_2.compile(optimizer='rmsprop',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    results_list.append(k_time_cross_validation(model_2, 'model2', raw_data))

    model_3 = keras.Sequential([
        keras.layers.Dense(INPUT_SIZE, activation=tf.nn.relu),
        keras.layers.Dense(60, activation=tf.nn.relu),
        keras.layers.Dense(20, activation=tf.nn.softmax)
    ])

    model_3.compile(optimizer='rmsprop',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    results_list.append(k_time_cross_validation(model_3, 'model3', raw_data))
    # ======================SGD WITH MOMENTUM======================================
    model_4 = keras.Sequential([
        keras.layers.Dense(INPUT_SIZE, activation=tf.nn.relu),
        keras.layers.Dense(130, activation=tf.nn.relu),
        keras.layers.Dense(20, activation=tf.nn.softmax)
    ])

    sgd_1 = keras.optimizers.SGD(momentum=0.9)

    model_4.compile(optimizer=sgd_1,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    results_list.append(k_time_cross_validation(model_4, 'model4_sgd', raw_data))

    model_5 = keras.Sequential([
        keras.layers.Dense(INPUT_SIZE, activation=tf.nn.relu),
        keras.layers.Dense(200, activation=tf.nn.relu),
        keras.layers.Dense(20, activation=tf.nn.softmax)
    ])

    model_5.compile(optimizer=sgd_1,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    results_list.append(k_time_cross_validation(model_5, 'model5_sgd', raw_data))

    model_6 = keras.Sequential([
        keras.layers.Dense(INPUT_SIZE, activation=tf.nn.relu),
        keras.layers.Dense(60, activation=tf.nn.relu),
        keras.layers.Dense(20, activation=tf.nn.softmax)
    ])

    model_6.compile(optimizer=sgd_1,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    results_list.append(k_time_cross_validation(model_6, 'model6_sgd', raw_data))

    print(results_list)
