import  numpy as np
import scipy.io as scio


def LR_train(x,y, lamb=0):
    X = np.c_[x, np.ones(x.shape[0])]
    w = y * np.matrix(np.dot(X, X.T) + lamb * np.identity(X.shape[0])).I * X
    return w
def LR_test(x, w):
    X = np.c_[x, np.ones(x.shape[0])]
    Y = np.dot(X, w.T)
    return Y

train_path = 'C:/Users/Administrator/Downloads/StellarSLOANDR7.zip/StellarSLOANDR7/StellarSLOANDR7Train.mat'
test_path = 'C:/Users/Administrator/Downloads/StellarSLOANDR7.zip/StellarSLOANDR7/StellarSLOANDR7Test.mat'

train_mat = scio.loadmat(train_path)
train_x = train_mat.get('train_x')
train_y = train_mat.get('train_y')[:, 2] # Choise one column as the label of data.
test_mat = scio.loadmat(test_path)
test_x = test_mat.get('test_x')
test_y = test_mat.get('test_y')[:, 2]

# Train
w = LR_train(train_x, train_y, 0.001)

# Test
y = LR_test(test_x, w)
