import tensorflow as tf
from tensorflow import keras
from read_data.read_xls import read_xls_ziwm
import numpy as np
from sklearn.model_selection import train_test_split


raw_data = read_xls_ziwm('/home/bomba/PycharmProjects/ZIwM/data/bialaczka.XLS') #All rows are 410
lables = []
rows = []
test_lables = []
test_rows = []

# l_raw_data, test_raw_data = tf.spli(tf.random_shuffle(raw_data))

ttrain, ttest = train_test_split(raw_data, test_size=0.2, random_state=42)

for index, row in raw_data.iterrows():
    if index < 310:
        lables.append(row['Unnamed: 0'])
        rows.append([i for i in row[2:22]])
    else:
        test_lables.append(row['Unnamed: 0'])
        test_rows.append([i for i in row[2:22]])

print(len(lables))
print(len(rows))
print(len(test_lables))
print(len(test_rows))


model = keras.Sequential([
    keras.layers.Dense(20),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(20, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

x = np.array(rows)
y = np.array(lables)

model.fit(x, y, epochs=5)

test_x = np.array(test_rows)
test_y = np.array(test_lables)

test_loss, test_acc = model.evaluate(test_x, test_y)
print(test_acc)
