import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


class LogisticRegression(object):
    def __init__(self, data, labels):
        """
        :param data: data 'N x number of features'
        :param labels: class one vs all type 'N x number of classes'
        """
        self.x = data
        self.y = labels
        self.W = np.random.random((self.x.shape[1], self.y.shape[1]))
        self.b = np.random.random(self.y.shape[1])

    def train(self, lr=0.1, data=None, L2_reg=0.00):
        """
        
        :param lr: Learning Rate 
        :param data: Input data to be trained
        :param L2_reg: Regularization (optional)
        """
        if data is not None:
            self.x = data

        d_y = self.y - sigmoid(np.dot(self.x, self.W) + self.b)

        self.W += lr * np.dot(self.x.T, d_y) - lr * L2_reg * self.W
        self.b += lr * np.mean(d_y, axis=0)

    def xentropy(self):
        """
        :return: The cross entropy 
        """
        sigmoid_activation = sigmoid(np.dot(self.x, self.W) + self.b)
        cross_entropy = - np.mean(np.sum(self.y * np.log(sigmoid_activation)+(1 - self.y) * np.log(1 - sigmoid_activation),axis=1))
        return cross_entropy

    def predict(self, x):
        """
        Calculates the probabilities of the data sample to belong to each class
        :param x: Data to be tested
        :return: A vector containing the probabilities of belonging to each class
        """
        return sigmoid(np.dot(x, self.W) + self.b)

    def plot_lr(self):
        xx, yy = np.mgrid[0:5:.1, 0:5:.1]
        probs = np.zeros(xx.shape)
        for k in range(3):
            for i in range(xx.shape[0]):
                for j in range(xx.shape[1]):
                    probs[i, j] = self.predict([xx[i, j], yy[i, j]])[k]
            plt.contour(xx, yy, probs, levels=[0.5])
            

def test_lr(learning_rate=0.01, n_epochs=20000, regularization=0.1):
    """
    A test function to check the working of the code composed
    :param learning_rate: The learning rate of the gradient descent
    :param n_epochs: number of iterations
    :param regularization: The amount of Regularization to be added 
    :return: none just plots
    """
    x = np.zeros((300, 2))
    x[0:100, :] = np.random.random((100, 2))
    x[100:200, :] = np.random.random((100, 2)) + 1
    x[200:300, :] = np.array([np.random.random((100)), np.random.random((100)) + 1]).T
    y = np.zeros((300, 3))
    y[0:100, :] = np.hstack((np.ones((100, 1)), np.zeros((100, 1)), np.zeros((100, 1))))
    y[100:200, :] = np.hstack((np.zeros((100, 1)), np.ones((100, 1)), np.zeros((100, 1))))
    y[200:300, :] = np.hstack((np.zeros((100, 1)), np.zeros((100, 1)), np.ones((100, 1))))

    classifierWreg = LogisticRegression(data=x, labels=y)

    # train
    for epoch in range(n_epochs):
        classifierWreg.train(lr=learning_rate, L2_reg=regularization)
        cost_reg = classifierWreg.xentropy()
        print(epoch, cost_reg)
    classifierWreg.plot_lr()
    plt.scatter(x[0:100, 0], x[0:100, 1])
    plt.scatter(x[100:200, 0], x[100:200, 1])
    plt.scatter(x[200:300, 0], x[200:300, 1])
    plt.show()


if __name__ == "__main__":
    test_lr()
