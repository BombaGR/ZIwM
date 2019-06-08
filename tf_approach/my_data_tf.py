import tensorflow as tf
from tensorflow import keras
from read_data.read_xls import read_xls_ziwm
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


def split_data(data, test_size=0.4, random_state=42):
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    frames = [train, test]
    return pd.concat(frames)


raw_data = read_xls_ziwm('/home/bomba/PycharmProjects/ZIwM/data/bialaczka.XLS')  # All rows are 410
lables = []
rows = []
test_lables = []
test_rows = []


for index, row in split_data(raw_data).iterrows():
    if len(lables) < 310:
        lables.append(row['Unnamed: 0'])
        rows.append([i / 8 for i in row[2:22]])
    else:
        test_lables.append(row['Unnamed: 0'])
        test_rows.append([i / 8 for i in row[2:22]])


model = keras.Sequential([
    keras.layers.Dense(20, activation=tf.nn.relu),
    keras.layers.Dense(130, activation=tf.nn.relu),
    keras.layers.Dense(20, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

x = np.array(rows)

y = np.array([i - 1 for i in lables])
print(y)
model.fit(x, y, epochs=400)

test_x = np.array(test_rows)

test_y = np.array([i - 1 for i in test_lables])
print(test_y)
test_loss, test_acc = model.evaluate(test_x, test_y)
