import numpy as np


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # Initialise the list of weight matrices, network architecture and learning rate
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # Start looping from the index of the first layer but stop before we reach the last 2 layers
        for i in np.arange(0, len(layers) - 2):
            # Randomy initialise a weight matrix connecting the number of nodes in each respective layer together,
            # adding an extra node for the bias
            w = np.random.randn(layers[i] + 1, layers[i+1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        # The last 2 layers are a special case where the input connections need a bias term but the output does not
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # Return string that represents the network architecture
        return 'Neural Network: {}'.format('-'.join(str(l) for l in self.layers))
    
    def sigmoid(self, x):
        return 1.0/(1 + np.exp(-x))
    
    def sigmoid_deriv(self, x):
        return x * (1 - x)
    
    def fit(self, X, y, epochs=1000, displayUpdate=100):
        X = np.c_[X, np.ones(X.shape[0])]
        
        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)
                
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))
    
    def fit_partial(self, x, y):
        A = [np.atleast_2d(x)]
        
        #FEEDFORWARD
        # loop over the layers in the network
        for layer in np.arange(0, len(self.W)):
            net = A[layer].dot(self.W[layer])
            out = self.sigmoid(net)
            A.append(out)
        
        #BACKPROPAGATION
        # the first phase of backpropagation is to compute the
        # difference between our *prediction* (the final output
        # activation in the activations list) and the true target
        # value
        error = A[-1] - y
        D = [error * self.sigmoid_deriv(A[-1])]
        for layer in np.arange(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)
        
        
        D = D[::-1]
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])
            
            
        
    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)
        
        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]
        
        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))
            
        return p
    
    def calculate_loss(self, X, target):
        targets = np.atleast_2d(target)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)
        
        return loss
        