import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from .logistic1d import evaluate_model,evaluate_predictions,apply_model
import numpy as np

def visualizar_superficie_error(ax_surface,x,y,m,b,error):
    ax_surface.set_xlabel("m")
    ax_surface.set_ylabel("b")
    ax_surface.set_zlabel("E")
    ax_surface.set_title("Superficie de E(m,b) ")
    ax_surface.set_zticks([])
    samples=100
    param_range=10
    M = np.linspace(-param_range, param_range, samples)
    B = np.linspace(-param_range, param_range, samples)    
    ms, bs = np.meshgrid(M, B)
    es=np.zeros_like(ms)
    n_m=ms.shape[0]
    n_b=ms.shape[0]
    
    for i in range(n_m):
        for j in range(n_b):
            es[i,j]=evaluate_model(x,y,ms[i,j],bs[i,j])
    
    surf = ax_surface.plot_surface(ms,bs,es, cmap=cm.coolwarm,alpha=0.5,linewidth=0, antialiased=False)
    ax_surface.scatter([m],[b],[error*1.1],c="green",s=70)
    plt.colorbar(surf, shrink=0.5, aspect=5)    
    
def visualizar_leyendas(ax_data,m,b,error):
    model = patches.Patch(color='red', label='Modelo: y=$\sigma$(x*{:.2f}+{:.2f})'.format(m,b))
    label='$E = $  %.2f' % (error)
    error_patch = patches.Patch(color='black', label=label)
    handles=[model,error_patch]
    
    ax_data.legend(handles=handles,fontsize=8)
            
def plot_regresion_logistica1d(x,y,m,b,x_label,y_label,title=""):
    # Visualizacion
    figure=plt.figure(figsize=(10,5),dpi=100)
    ax_data=figure.add_subplot(1,2,1)
    ax_surface=figure.add_subplot(1,2,2,projection='3d')
    
    #dibujar datos
    ax_data.scatter(x,y,color="blue")
        
    #Etiquetas y titulos
    ax_data.set_xlabel(x_label)
    ax_data.set_ylabel(y_label)
    ax_data.set_title(title)
    
    #Dibujar f logística
    x_pad=40
    min_x,max_x=x.min()-x_pad,x.max()+x_pad
    x_plot=np.linspace(min_x,max_x,40)
    y_plot=apply_model(x_plot,m,b)
    ax_data.plot(x_plot,y_plot,'-')
    error=evaluate_model(x,y,m,b)
    # Mostrar leyendas
    visualizar_leyendas(ax_data,m,b,error)
    visualizar_superficie_error(ax_surface,x,y,m,b,error)
    
    plt.show()