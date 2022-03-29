import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))
def linear(x,m,b):
    return x*m+b

def forward(x,m,b):
    return sigmoid(linear(x,m,b))

def mean_binary_cross_entropy(y,yhat):
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
    return errors.mean()
    
def evaluate_model(x,y,m,b):
    yhat=forward(x,m,b)
    return mean_binary_cross_entropy(y,yhat)