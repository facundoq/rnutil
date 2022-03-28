import numpy as np

def equals_vector(x, y):
    return np.all(x==y)

def verificar_igualdad(x,y):
    iguales=equals_vector(x, y)
    if iguales:
        print("Los vectores x e y son iguales:")
    else:
        print("Los vectores x e y son distintos:")

    print("x: ", x)
    print("y: ", y)
