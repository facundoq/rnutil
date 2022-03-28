import numpy as np

def logistic(x):
    return 1/(1+np.exp(-x))
def linear(x,m,b):
    return x*m+b

def apply_model(x,m,b):
    return logistic(linear(x,m,b))

def evaluate_predictions(y,yhat):
    eps = np.finfo(float).eps
    n=len(y)
    errors=np.zeros(n)
    for i in range(n):
        if y[i]==1:
            errors[i]=-np.log(yhat[i]+eps)
        else: #y[i]==0
            errors[i]=-np.log(1-yhat[i]+eps)
            
    # implementación vectorial
    #errors=y* (-log(yhat)) +(1-y)* (-log(-yhat))
    return errors.sum()
    
def evaluate_model(x,y,m,b):
    yhat=apply_model(x,m,b)
    return evaluate_predictions(y,yhat)