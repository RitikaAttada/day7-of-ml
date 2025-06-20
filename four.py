import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras

def prob(num1, num2):
    mp = 'four.keras'
    x = np.linspace(-5, 5, 100)
    z = np.linspace(-5, 5, 100)
    X,Z = np.meshgrid(x,z)
    X = X.flatten()
    Z = Z.flatten()
    Y = np.cos(X*Z)
    xn = X.min()
    xx = X.max()
    zn = Z.min()
    zx = Z.max()
    yn = Y.min()
    yx = Y.max()
    xnorm = 2*(X-xn)/(xx - xn)-1
    znorm = 2*(Z-zn)/(zx - zn)-1
    Y = (Y-yn)/(yx - yn)
    inp = np.column_stack((xnorm,znorm))
    if (os.path.exists(mp)):
        model = keras.models.load_model(mp)
    else:
        model = keras.Sequential([
            keras.layers.Dense(64, activation='tanh', input_shape=(2,)),
            keras.layers.Dense(64, activation='tanh'),
            keras.layers.Dense(32, activation='tanh'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer = 'Adam', loss='mean_squared_error')
        h = model.fit(inp, Y, epochs=500, batch_size = 128)
        lv = h.history['loss']
        pred = model.predict(inp)
        model.save('four.keras')
        plt.figure()
        plt.title('loss values')
        plt.plot(lv)
        plt.grid(True)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()
        plt.figure()
        plt.scatter(Y, pred, label='actual vs predicted', color='green')
        plt.grid(True)
        plt.xlabel('actual')
        plt.ylabel('predicted')
        plt.legend()
        plt.show()
    return model.predict(np.array([[2*(num1 - xn)/(xx - xn) -1, 2*(num2-zn)/(zx - zn)-1]]))[0][0]*(yx-yn)+yn

print(prob(0,0))
print(prob(np.pi,1))
print(prob(2,2))

