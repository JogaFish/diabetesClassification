import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


def build_model():
    dnn_model = Sequential([
        Dense(units=500, input_dim=8, activation='sigmoid'),
        Dense(units=100, activation='sigmoid'),
        Dense(units=2, activation='softmax')
    ])
    dnn_model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(0.001),
                      metrics=['accuracy'])
    return dnn_model


url = 'diabetes.csv'
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                'DiabetesPedigreeFunction', 'Age', 'Outcome']
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
            'DiabetesPedigreeFunction', 'Age']

x = pd.read_csv(url, names=column_names)

x = x.drop(0)
y = x.pop('Outcome')

for col in features:
    x[col] = pd.to_numeric(x[col])

y = pd.to_numeric(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=40)

model = build_model()
model.fit(x_train, y_train, epochs=20, batch_size=70)

scores_train = model.evaluate(x_train, y_train, verbose=0)
print("Accuracy for training data " + str(scores_train[1]))

scores_test = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy for test data " + str(scores_test[1]))




