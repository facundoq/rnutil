import os
from pathlib import Path
import numpy as np
import pandas as pd
from skimage import io



_ROOT = os.path.abspath(os.path.dirname(__file__))

def data_path(local=False):
    if local:
        path = Path().absolute()
        print(f"Loading file from current directory ({path})...")
        return path
    else:
        module_name = __name__.split(".")[0]
        path = Path(_ROOT)/'data'
        print(f"Loading file from package {module_name} ({path})...")
        return path
    

def load_dataset_numpy(filename,local=False):
    path = data_path(local)/filename
    result= np.loadtxt(path, delimiter=",", skiprows=1)
    print("Done")
    return result

def load_dataset_pandas(filename,local=False):
    result= pd.read_csv(data_path(local)/filename)
    print("Done")
    return result

def load_image(filename,local=False):
    result= io.imread(data_path(local)/filename)
    print("Done")
    return result



# dividir dataset en training y testing de forma aleatoria    
def dividir_train_test(X, Y, test_size= 0.2):
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= test_size)
    return X_train, X_test, Y_train, Y_test
