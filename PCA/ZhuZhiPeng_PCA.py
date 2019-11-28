import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

data = pd.read_csv('./train.csv', header=0)
imgs = data.iloc[1:, 1:].values
labels = data.iloc[1:, 0].values

nn = MLPClassifier(hidden_layer_sizes=(32, ), activation='relu', solver='sgd')
np.random.seed(666)
pca = PCA(n_components=5)
imgs = pca.fit_transform(imgs)
X_train, X_test, Y_train, Y_test = train_test_split(
        imgs, labels, test_size=0.33, random_state=23323)
nn.fit(X_train, Y_train)

# Predict
result_y = nn.predict(X_test)
correct = (Y_test == result_y).sum()
correct_rate = correct / Y_test.size
print('Correct rate: %.2f%%' % (correct_rate * 100))