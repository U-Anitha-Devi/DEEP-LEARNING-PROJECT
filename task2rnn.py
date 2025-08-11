import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Input
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
time_steps = np.linspace(0, 100, 1000)
data = np.sin(time_steps)
scaler = MinMaxScaler()
data = scaler.fit_transform(data.reshape(-1, 1))
X = []
y = []
timesteps = 10
for i in range(len(data) - timesteps):
    X.append(data[i:i+timesteps])
    y.append(data[i+timesteps])
X = np.array(X)
y = np.array(y)
model = Sequential([
    Input(shape=(timesteps, 1)),
    SimpleRNN(50, activation='tanh'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=32)
predicted = model.predict(X)
plt.plot(y, label='True')
plt.plot(predicted, label='Predicted')
plt.legend()
plt.show()





