from nn import Perceptron
import numpy as np

# construt the OR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # 4x2
y = np.array([[0], [1], [1], [1]])             # 4x1

print("[INFO] training process..")

p = Perceptron(X.shape[1], alpha=0.1) #X.shape[1] = 2
p.fit(X, y, epochs=20)

print("[INFO] testing perceptron...")

for (x, target) in zip(X, y):
    pred = p.predict(x)
    
    print("[INFO] data={}, ground-truth={}, pred={}".format(x, target[0], pred))