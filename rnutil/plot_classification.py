import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.patches as patches

# imprime los puntos para un dataset bidimensional junto con la frontera de decisión del modelo
def plot_fronteras_keras(modelo, x, y,title="",detail=0.1,padding=10):

    assert x.shape[1]==2,f"x debe tener solo dos variables de entrada (tiene {x.shape[1]})"
    # nueva figura
    plt.figure()
    # gráfico con la predicción aprendida
    x_min, x_max = x[:, 0].min() - padding, x[:, 0].max() + padding
    y_min, y_max = x[:, 1].min() - padding, x[:, 1].max() + padding
    xx, yy = np.meshgrid(np.arange(x_min, x_max, detail),
                         np.arange(y_min, y_max, detail))

    Z = np.c_[xx.ravel(), yy.ravel()]

    Z = modelo.predict(Z)
    Z = Z.argmax(axis=1)  # para Keras
    titulo = f"{title}: regiones de cada clase"
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.7,cmap='coolwarm')  # ,  cmap='RdBu')
    plt.colorbar()
    plt.title(titulo)

    # puntos con las clases
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()


# imprime los puntos para un dataset bidimensional junto con la frontera de decisión del modelo
def plot_clasificador_2D(modelo, x, y,title="",detail=0.1):

    assert x.shape[1]==2,f"x debe tener solo dos variables de entrada (tiene {x.shape[1]})"
    # nueva figura
    plt.figure()
    # gráfico con la predicción aprendida
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, detail),
                         np.arange(y_min, y_max, detail))

    Z = np.c_[xx.ravel(), yy.ravel()]

    Z = modelo.predict(Z)
    Z = Z.argmax(axis=1)  # para Keras
    titulo = f"{title}: regiones de cada clase"
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)  # ,  cmap='RdBu')
    plt.colorbar()
    plt.title(titulo)


    # puntos con las clases
    plt.scatter(x[:, 0], x[:, 1], c=y)    