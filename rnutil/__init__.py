
print(f"redes-neuronales-util: Los archivos de los conjuntos de datos que ofrece este paquete también están disponibles en https://github.com/facundoq/redes-neuronales-util/tree/main/rnutil/data para descargar de forma individual")
print("redes-neuronales-util: Agregando el parámetro local=True en las funciones rnutil.load_dataset_numpy, rnutil.load_dataset_pandas y rnutil.load_image se puede cargar una versión local de un archivo en lugar de la versión que ofrece este paquete.")

from .datasets import load_dataset_numpy,load_dataset_pandas,load_image

from .numpy import equals_vector,verificar_igualdad

from .plot import plot_regresion_lineal,plot_regresion_lineal_univariada,plot_regresion_logistica2D,plot_loss