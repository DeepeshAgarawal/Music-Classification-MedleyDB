import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.mlab as mlab
import timeit

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'


def plot_model(model, data):
    """

    :param model: the GMM model
    :param data: the data set 2D
    :return:
    """
    delta = 0.025
    x = np.arange(0.0, 4, delta)
    y = np.arange(0.0, 4, delta)
    X, Y = np.meshgrid(x, y)
    z = np.zeros((np.size(x), np.size(y)))
    # sum of Gaussian
    plt.figure()
    for i in range(np.size(model)):
        ztemp = mlab.bivariate_normal(X, Y, np.sqrt(model['cov'][i][0, 0]), np.sqrt(model['cov'][i][1, 1]), model['mu'][i][0], model['mu'][i][1], model['cov'][i][0,1])
        plt.contour(X, Y, model['w'][i]*ztemp)
        z = np.add(z, ztemp)
    plt.scatter(data[0, :], data[1, :], s=5)
    plt.figure()
    plt.contour(X, Y, z*np.size(model))
    plt.scatter(data[0, :], data[1, :], s=5)


def pdf_multivariate_gauss(x, w, mu, cov):
    """

    :param x: Feature Vector
    :param w: Weight
    :param mu: Mean
    :param cov: Covariance
    :return: the value of the PDF
    """

    weight = 1 / (((2 * np.pi) ** (len(mu) / 2)) * (np.linalg.det(cov)) ** (1 / 2))
    power_term = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov)).dot((x - mu)))
    return float(w * weight * np.exp(power_term))


def weight(x, w, mu, cov):
    return float(w / (((2 * np.pi) ** (len(mu) / 2)) * (np.linalg.det(cov)) ** (1 / 2)))


def power_term(x, w, mu, cov):
    power = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov)).dot((x - mu)))
    return float(power)

def gmm_init(nom, d, data):
    """

    :param nom: number of Gaussian models
    :param d: dimension of the feature vector
    :param data: Dataset d x Samples
    :return : Structure with
                N items
                'w'   - '1 x 1' - weight of the gaussian pdf
                'mu'  - 'd x 1' - mean --------"------------
                'cov' - 'd x d' - cov matrix -------"-------
    """
    model = np.zeros(nom, dtype=[('w',float),
                           ('mu', float, d),
                           ('cov', float, (d, d))])
    model['w'] = 1/nom
    # model['mu'] = np.random.random((nom, d))
    # model['cov'] = np.random.random((nom, d, d))
    for i in range(nom):
        # model['mu'][i] = np.random.randint(0,5,np.size(data, 0))
        model['mu'][i] = data[:, np.int(np.random.random()*np.size(data, 1))]
        model['cov'][i] = npm.eye(d)
    return model


def gmm_fit(data, model, runs=50, plot = 0, progress = 0):
    """

    :param data: data points 'd x N' where 'd' number of features ; 'N' number of samples
    :param model: the structure which contains the gmm mean and cov ie x from gmm_init
    :param runs: number of times the model updates
    :param plot:
    :param progress:
    :return:
    """
    d = np.size(data, 0)
    N = np.size(data, 1)
    nom = np.size(model)

    model_update = np.zeros(nom, dtype=[('w', float), ('mu', float, d), ('cov', float, (d, d))])

    if plot == 1:
        plot_model(model)
        plt.scatter(data[0, :], data[1, :], s=5)
    est=np.zeros(nom)
    power=np.zeros(nom)

    for k in range(runs):
        sumpx = np.zeros(nom)
        munum = np.zeros((d, nom))
        covnum = np.zeros((d, d, nom))
        for i in range(N):
            for j in range(nom):
                power[j] = power_term(data[:, i], model['w'][j], model['mu'][j], model['cov'][j])
            max_power = np.amax(power)
            for j in range(nom):
                est[j] = weight(data[:, i], model['w'][j], model['mu'][j], model['cov'][j])*np.exp(power[j]-max_power)
            fx = np.sum(est)
            px = est/fx
            sumpx = np.add(sumpx, px)
            for l in range(nom):
                munum[:, l] = np.add(munum[:, l], data[:, i] * px[l])
        model_update['mu'] = (munum / sumpx).T
        for i in range(N):
            for j in range(nom):
                power[j] = power_term(data[:, i], model['w'][j], model['mu'][j], model['cov'][j])
            max_power = np.amax(power)
            for j in range(nom):
                est[j] = weight(data[:, i], model['w'][j], model['mu'][j], model['cov'][j])*np.exp(power[j]-max_power)
            fx = np.sum(est)
            px = est/fx
            for l in range(nom):
                covnum[:, :, l] = np.add(covnum[:, :, l], px[l] * np.outer(data[:, i] - model_update['mu'][l], data[:, i] - model_update['mu'][l]))
                # covnum[:, :, l] = np.add(covnum[:, :, l], np.diag(np.diag(px[l] * np.outer(data[:, i] - model_update['mu'][l], data[:, i] - model_update['mu'][l]))))
        model_update['w'] = sumpx/N
        # model_update['mu'] = (munum/sumpx).T
        model_update['cov'] = (covnum/sumpx).T
        # for l in range(nom):
        #     model_update_2['w'][l] = sumpx[l]/N
        #     model_update_2['mu'][l] = munum[:, l]/sumpx[l]
        #     model_update_2['cov'][l] = covnum[:, :, l]/sumpx[l]
        for m in range(nom):
            det = np.linalg.det(model_update['cov'][m])
            if det == 0:
                model_update['cov'][m] = npm.eye(d)
                model_update['mu'][m] = data[:, np.int(np.random.random()*np.size(data, 1))]
        model = model_update
        if plot == 1:
            plot_model(model)
            plt.scatter(data[0, :], data[1, :], s=5)
        if progress == 1:
            print((k/runs)*100)
    return model

k = 1000
j = 10
data = np.zeros((2, k))
for i in range(j):
    data[:, i*np.int(k/j):(i+1)*np.int(k/j)] = np.random.random()*2 + np.random.random((2,np.int(k/j)))*np.random.random()

model = gmm_init(10, 2, data)
# plot_model(model,data)
start = timeit.default_timer()
for i in range(1):
    if i == 0:
        fitted = gmm_fit(data, model, 50, progress=1)
    else:
        fitted = gmm_fit(data, fitted, 10, progress=1)
    plot_model(fitted, data)
    plt.show()
stop = timeit.default_timer()
print(stop - start)
print(fitted)
plot_model(fitted, data)
plt.show()