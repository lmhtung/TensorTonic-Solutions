import numpy as np
    
def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))
    
def binary_cross_entropy_loss(p, y):
    n = y.shape[0]
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1 - epsilon)
    loss = (-1/n) * np.sum(y*np.log(p) + (1-y)*np.log(1-p))
    return loss
        
def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    n, d = X.shape
    w = np.random.randn(d) * 1e-2
    b = 1e-10
    y = y.reshape(n)
            
    for i in range(steps):
        # forward
        z = np.dot(X, w) + b
        p = _sigmoid(z)
    
        # loss
        loss = binary_cross_entropy_loss(p, y)
    
        # gradient
        dz = p - y
        dl_dw = (1/n) * np.dot(X.T, dz)
        dl_db = (1/n) * np.sum(dz)
    
        # update w, b
        w -= lr*dl_dw
        b -= lr*dl_db
            
    return w, b
    pass