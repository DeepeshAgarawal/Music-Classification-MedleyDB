import numpy as np
import matplotlib.pyplot as plt
from random import sample


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def softmax(x):
    e = np.exp(x - np.max(x))
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2


class LogisticRegression(object):
    def __init__(self, data, labels):
        """
        :param data: data 'N x number of features'
        :param labels: class one vs all type 'N x number of classes'
        """
        self.x = data
        self.y = labels
        self.W = np.zeros((self.x.shape[1], self.y.shape[1]))
        self.b = np.zeros(self.y.shape[1])

    def train(self, lr=0.1, data=None, L2_reg=0.00):
        """
        
        :param lr: Learning Rate 
        :param data: Input data to be trained
        :param L2_reg: Regularization (optional)
        """
        if data is not None:
            self.x = data

        d_y = self.y - softmax(np.dot(self.x, self.W) + self.b)

        self.W += lr * np.dot(self.x.T, d_y) - lr * L2_reg * self.W
        self.b += lr * np.mean(d_y, axis=0)

    def xentropy(self):
        """
        :return: The cross entropy 
        """
        sigmoid_activation = softmax(np.dot(self.x, self.W) + self.b)
        cross_entropy = - np.mean(np.sum(self.y * np.log(sigmoid_activation)+(1 - self.y) * np.log(1 - sigmoid_activation),axis=1))
        return cross_entropy

    def predict(self, x):
        """
        Calculates the probabilities of the data sample to belong to each class
        :param x: Data to be tested
        :return: A vector containing the probabilities of belonging to each class
        """
        return softmax(np.dot(x, self.W) + self.b)

    def plot_lr(self):
        xx, yy = np.mgrid[0:5:.1, 0:5:.1]
        probs = np.zeros(xx.shape)
        for k in range(3):
            for i in range(xx.shape[0]):
                for j in range(xx.shape[1]):
                    probs[i, j] = self.predict([xx[i, j], yy[i, j]])[k]
            plt.contour(xx, yy, probs, levels=[0.5])


table = np.genfromtxt('featuresNew.csv', delimiter=',')
# table = np.load('lda_norm.npy')
f = np.int(0.8 * table.shape[0])
indices = sample(range(table.shape[0]), f)
test = np.delete(table, indices, axis=0)
# train = np.load('train.npy')
# test  = np.load('test.npy')
temp = table[indices]
fvalid = np.int(0.2 * table.shape[0])
indices = sample(range(temp.shape[0]), fvalid)
validation = temp[indices]
train = np.delete(temp, indices, axis=0)
noc = 9
labels = np.zeros((train.shape[0], noc))
train_curve = np.zeros(np.arange(0.1, 1.01, 0.05).shape[0])
test_curve = np.zeros(np.arange(0.1, 1.01, 0.05).shape[0])
val_curve = np.zeros(np.arange(0.1, 1.01, 0.05).shape[0])
k = -1
noi = 10000
learning_rate = 0.0015
regularization = 20
for percent in np.arange(0.1, 1.01, 0.05):
    k += 1
    tempind = sample(range(train.shape[0]), np.int(percent * train.shape[0]))
    temp_train = train[tempind]

    temp_labels = np.zeros((temp_train.shape[0], noc))

    for i in range(temp_train.shape[0]):
        temp_labels[i, np.int(np.array(temp_train[i, temp_train.shape[1]-1]))] = 1

    classifierWreg = LogisticRegression(data=temp_train[:, :temp_train.shape[1]-1], labels=temp_labels)
    print(percent)

    for iteration in range(noi):
        classifierWreg.train(lr=learning_rate,L2_reg=regularization)
        cost_reg = classifierWreg.xentropy()
        # print(iteration, cost_reg)

    #     prediction
    correctPred = 0
    totalTested = 0

    for a, b in zip(temp_train[:, :temp_train.shape[1]-1], temp_labels):
        predValue=classifierWreg.predict(a)
        indexMax = np.argmax(predValue)
        if indexMax == np.argmax(b):
            correctPred += 1
        totalTested += 1

    train_curve[k] = correctPred/totalTested
    print(train_curve[k])

    correctPred = 0
    totalTested = 0

    for a, b in zip(validation[:, :validation.shape[1] - 1], validation[:, validation.shape[1] - 1] ):
        predValue = classifierWreg.predict(a)
        indexMax = np.argmax(predValue)
        if indexMax == b:
            correctPred += 1
        # print predValue,indexMax,b
        totalTested += 1

    val_curve[k] = correctPred / totalTested
    print(val_curve[k])

    # correctPred = 0
    # totalTested = 0

    # for a, b in zip(test[:, :test.shape[1] - 1], test[:, test.shape[1] - 1] ):
    #     predValue = classifierWreg.predict(a)
    #     indexMax = np.argmax(predValue)
    #     if indexMax == b:
    #         correctPred += 1
    #     # print predValue,indexMax,b
    #     totalTested += 1
    #
    # test_curve[k] = correctPred / totalTested
    # print(test_curve[k])

np.save('LogRegClassifier.npy', classifierWreg)
correctPred = 0
totalTested = 0

for a, b in zip(test[:, :test.shape[1] - 1], test[:, test.shape[1] - 1] ):
    predValue = classifierWreg.predict(a)
    indexMax = np.argmax(predValue)
    if indexMax == b:
        correctPred += 1
    # print predValue,indexMax,b
    totalTested += 1

test_curve[k] = correctPred / totalTested
print("test accuracy", test_curve[k])

plt.plot(np.arange(0.1, 1.01, 0.05), train_curve*100, label="Train Accuracy")
plt.plot(np.arange(0.1, 1.01, 0.05), val_curve*100, label="Validation Accuracy")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.xlabel('Data used for Training (in %)')
plt.ylabel('Accuracy(in %)')
plt.show()
