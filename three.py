import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras

def prob(num1, num2):
    mp = 'three.keras'
    x = np.linspace(0,100,100)
    z = np.linspace(0,100,100)
    X,Z = np.meshgrid(x,z)
    X = X.flatten()
    Z = Z.flatten()
    Y = np.abs(X-Z)
    xn = X.min()
    xx = X.max()
    zn = Z.min()
    zx = Z.max()
    yn = Y.min()
    yx = Y.max()
    xnorm = (X-xn)/(xx - xn)
    znorm = (Z-zn)/(zx - zn)
    ynorm = (Y-yn)/(yx - yn)
    inp = np.column_stack((xnorm,znorm))
    if (os.path.exists(mp)):
        model = keras.models.load_model(mp)
    else:
        model = keras.Sequential([keras.layers.Dense(32,input_shape=(2,),activation='tanh'),
                                 keras.layers.Dense(16, activation='tanh'),
                                 keras.layers.Dense(units=1)])
        model.compile(optimizer = 'Adam', loss='mean_squared_error')
        h = model.fit(inp, ynorm, epochs=100)
        lv = h.history['loss']
        pred = model.predict(inp)
        model.save('three.keras')
        plt.figure()
        plt.title('loss values')
        plt.plot(lv)
        plt.grid(True)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()
        plt.figure()
        plt.scatter(ynorm, pred, label='actual vs predicted', color='green')
        plt.grid(True)
        plt.xlabel('actual')
        plt.ylabel('predicted')
        plt.legend()
        plt.show()
    return model.predict(np.array([[(num1 - xn)/(xx - xn), (num2-zn)/(zx - zn)]]))[0][0]*(yx-yn)+yn

print(prob(50,50))
print(prob(75,25))
print(prob(10,90))

