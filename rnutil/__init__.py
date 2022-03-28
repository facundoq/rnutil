print("redes-neuronales-uba: Los archivos de los conjuntos de datos que ofrece este paquete también están disponibles en https://github.com/facundoq/redes-neuronales-util/rnutil/data para descargar de forma individual")
print("redes-neuronales-uba: Agregando el parámetro local=True en las funciones rnutil.load_dataset_numpy, rnutil.load_dataset_pandas y rnutil.load_image se puede cargar una versión local de un archivo en lugar de la versión que ofrece este paquete.")

from .datasets import load_dataset_numpy,load_dataset_pandas,load_image