import numpy as np
import gmm
from random import sample
# table -- N x nof+1


def create_conf_matrix(expected, predicted, n_classes):
    m = [[0] * n_classes for i in range(n_classes)]
    for pred, exp in zip(predicted, expected):
        m[pred][exp] += 1
    return m


def calc_accuracy(conf_matrix):
    t = sum(sum(l) for l in conf_matrix)
    return sum(conf_matrix[i][i] for i in range(len(conf_matrix))) / t

# table = np.genfromtxt('lda_norm.csv', delimiter=',')
# table[:, :table.shape[1]-1] = (table[:, :table.shape[1]-1] - np.amin(table[:, :table.shape[1]-1]))/(np.amax(table[:, :table.shape[1]-1])-np.amin(table[:, :table.shape[1]-1]))
# f is number of elements needed
table = np.load('lda_norm.npy')
f = np.int(0.8*table.shape[0])

indices = sample(range(table.shape[0]), f)
# train = np.load('train.npy')
# test  = np.load('test.npy')
train = table[indices]
test = np.delete(table, indices, axis=0)
np.save('train.npy', train)
np.save('test.npy', test)

train_labels, train_features = train[:, train.shape[1]-1], train[:, :train.shape[1]-1]
train_dic = {label: train_features[train_labels == label] for label in np.unique(train_labels)}

test_labels, test_features = test[:, test.shape[1]-1], test[:, :test.shape[1]-1]
test_dic = {label: test_features[test_labels == label] for label in np.unique(test_labels)}

# noc - number of classes
# nom - number of gaussians per label
# noi - number of iterations

noc = 9
nom = 5
noi = 5

# training

for i in range(noc):
    print(i)
    print(train_dic[i].shape)
    temp_model = gmm.gmm_init(nom, train_features.shape[1], train_dic[i].T)
    # fitted = gmm.gmm_init(nom, train_features.shape[1], train_dic[i].T)
    fitted = gmm.gmm_fit(train_dic[i].T, temp_model, noi, progress=0)
    if i == 0:
        gmm_model = fitted
    else:
        gmm_model = np.vstack((gmm_model, fitted))

# testing

est = np.zeros(nom)
predicted = np.zeros(train_labels.shape[0])
power = np.zeros(nom)
weight = np.zeros(nom)

for i in range(train_features.shape[0]):
    max_fx = 0
    for k in range(noc):
        for j in range(nom):
            power[j] = gmm.power_term(train_features.T[:, i], gmm_model[k]['w'][j], gmm_model[k]['mu'][j],gmm_model[k]['cov'][j])
            weight[j] = gmm.weight_term(train_features.T[:, i],gmm_model[k]['w'][j], gmm_model[k]['mu'][j],gmm_model[k]['cov'][j])
        norm_power = (power - np.amax(power))
        est = np.multiply(np.exp(norm_power), weight)
        fx = np.sum(est)
        if fx > max_fx:
            max_fx = fx
            predicted[i] = k
conf = create_conf_matrix(train_labels.astype(int), predicted.astype(int), np.int(noc))
acc = calc_accuracy(conf)
print(conf)
print(acc)

predicted = np.zeros(test_labels.shape[0])

for i in range(test_features.shape[0]):
    max_fx = 0
    for k in range(noc):
        for j in range(nom):
            power[j] = gmm.power_term(test_features.T[:, i], gmm_model[k]['w'][j], gmm_model[k]['mu'][j],gmm_model[k]['cov'][j])
            weight[j] = gmm.weight_term(test_features.T[:, i],gmm_model[k]['w'][j], gmm_model[k]['mu'][j],gmm_model[k]['cov'][j])
        norm_power = (power - np.amax(power))
        est = np.multiply(np.exp(norm_power), weight)
        fx = np.sum(est)
        if fx > max_fx:
            max_fx = fx
            predicted[i] = k

conf = create_conf_matrix(test_labels.astype(int), predicted.astype(int), np.int(noc))
acc = calc_accuracy(conf)
np.save('conf_mat.npy', conf)
print(conf)
print(acc)


# print(gmm_model)
np.save('gmm_model.npy', gmm_model)