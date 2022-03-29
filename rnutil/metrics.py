import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_training_curves(history):
    plt.figure()
    plot_training_curve(history,key="loss",own_figure=False)
    extra_keys = ["acc","accuracy","mae","mse"]
    for key in extra_keys:
        if key in history.history:
            plot_training_curve(history,key=key,own_figure=False)
    plt.xlabel('epoch')
    plt.ylabel('metrics')
    plt.legend()
        
# graficar curvas de entrenamiento
def plot_training_curve(history, key="loss",own_figure=True):
    if own_figure:
        plt.figure()
    # summarize history for loss
    plt.plot(history.history[key], label=f"Train {key}")
    val_key = f'val_{key}'
    if val_key in history.history:
        plt.plot(history.history, label=f"Val {val_key}")
        
    if own_figure:
        plt.title(f'model {key}')
        plt.ylabel(key)
        plt.legend()
        plt.xlabel('epoch')
    


def plot_loss(loss_history):
    plt.figure()
    epochs = np.arange(1,len(loss_history)+1)
    plt.plot(epochs,loss_history)
    plt.xlabel("Época")
    plt.ylabel("Error")
    

def plot_loss_accuracy(loss_history,acc_history):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    epochs = np.arange(1,len(loss_history)+1)
    ax1.plot(epochs,loss_history)
    ax2.plot(epochs,acc_history,c="orange")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Error")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim([0,1])
    plt.tight_layout()

def plot_loss_accuracy_keras(history):
    plot_loss_accuracy(history.history["loss"],history.history["accuracy"])


# Crea y Grafica una matriz de confusión
# PARAM:
#       real_target = vector con valores esperados
#       pred_target = vector con valores calculados por un modelo
#       classes = lista de strings con los nombres de las clases.
def plot_confusion_matrix(real_target, pred_target, classes=[],  normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    from sklearn.metrics import confusion_matrix
    import itertools
    if (len(classes)==0):
        classes= [str(i) for i in range(int(max(real_target)+1))]  # nombres de clases consecutivos
    cm= confusion_matrix(real_target, pred_target)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

#    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



# Grafica la curva ROC y la curva precision-recall para el modelo y los datos pasados como argumentos 
def plot_tradeoff_curves(modelo, x, y):
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
    # probabilidades para los datos
    assert np.all(np.logical_or(y==0,y==1)), "y debe ser un vector con etiquetas 0 o 1"
    
    y_score = modelo.predict(x)[:,1] # se queda con la clase 1
    # Create true and false positive rates
    false_positive_rate, true_positive_rate, threshold = roc_curve(y, y_score)
    # Precision-recall curve
    precision, recall, _ = precision_recall_curve(y, y_score)
    
    # ROC
    plt.figure()
    plt.title('ROC. Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, label='ROC curve (area = %0.2f)' % roc_auc_score(y, y_score))
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.legend(loc="lower right")
    plt.ylabel('True Positive Rate (Recall)')
    plt.xlabel('False Positive Rate (1- Especificidad)')
    plt.show()
    
    # precision-recall curve
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve')   
    
def print_classification_report(y_true, y_pred):
    from sklearn.metrics import f1_score,  recall_score, precision_score, accuracy_score
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    assert len(y_true.shape)==1, "y_true debe ser un vector 1D de etiquetas"
    if len(y_pred.shape)==2:
        # probs to class number
        y_pred = np.argmax(y_pred,axis = 1) 
    assert len(y_pred.shape)==1, "y_true debe ser un vector 1D de etiquetas"
    assert len(y_pred)==len(y_true), "La cantidad de ejemplos debe y_true y y_pred debe ser la misma"
    classes= max(y_true)+1
    
    print("   Accuracy: %.2f    (%d ejemplos)"  % (accuracy_score(y_true, y_pred), y_true.shape[0]))
    if (classes==2):
        print("  Precision: %.2f" % precision_score(y_true, y_pred) )
        print("     Recall: %.2f" % recall_score(y_true, y_pred ))
        print("  f-measure: %.2f" % f1_score(y_true, y_pred))
        
        