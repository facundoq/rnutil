
print(f"{__name__}: Los archivos de los conjuntos de datos que ofrece este paquete también están disponibles en https://github.com/facundoq/redes-neuronales-util/tree/main/rnutil/data para descargar de forma individual")
print(f"{__name__}: Agregando el parámetro local=True en las funciones {__name__}.load_dataset_numpy, {__name__}.load_dataset_pandas y {__name__}.load_image se puede cargar una versión local de un archivo en lugar de la versión que ofrece este paquete.")

from .datasets import load_dataset_numpy,load_dataset_pandas,load_image,dividir_train_test

from .numpy import equals_vector,verificar_igualdad

from .plot import get_categorical_colormap

from .plot_logistic import plot_regresion_logistica1d,plot_regresion_logistica2D 
from .plot_regression import plot_regression1D,plot_regresion_lineal,plot_regresion_lineal_univariada
from .plot_classification import plot_fronteras_keras

from .metrics import plot_loss,plot_loss_accuracy,plot_loss_accuracy_keras,plot_confusion_matrix,plot_tradeoff_curves,plot_training_curve,plot_training_curves,print_classification_report