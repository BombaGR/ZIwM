import tensorflow as tf
from tensorflow import keras
from read_data.read_xls import read_xls_ziwm
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

raw_data = read_xls_ziwm('/home/bomba/PycharmProjects/ZIwM/data/bialaczka.XLS')  # All rows are 410
lables = []
rows = []
test_lables = []
test_rows = []

ttrain, ttest = train_test_split(raw_data, test_size=0.4, random_state=42)

frames = [ttrain, ttest]

random_data = pd.concat(frames)

# tttrain, tttest = train_test_split(random_data, test_size=0.2)
#
# frames = [tttrain, tttest]
#
# random_data_2 = pd.concat(frames)

# print(len(raw_data))
# print(len(random_data))

for index, row in random_data.iterrows():
    if len(lables) < 310:
        lables.append(row['Unnamed: 0'])
        rows.append([i / 8 for i in row[2:22]])
    else:
        test_lables.append(row['Unnamed: 0'])
        test_rows.append([i / 8 for i in row[2:22]])

# print(len(lables))
# print(len(rows))
# print(len(test_lables))
# print(len(test_rows))


model = keras.Sequential([
    keras.layers.Dense(20, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(20, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

x = np.array(rows)

y = np.array([i-1 for i in lables])
print(y)
model.fit(x, y, epochs=400)

test_x = np.array(test_rows)


test_y = np.array([i-1 for i in test_lables])
print(test_y)
test_loss, test_acc = model.evaluate(test_x, test_y)
