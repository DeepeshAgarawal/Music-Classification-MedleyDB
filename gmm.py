import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.mlab as mlab
import timeit
import kmeans

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
    plt.contour(X, Y, z)
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


def weight_term(x, w, mu, cov):
    det = np.linalg.det(cov)
    if det < 10**-60:
        det = 10**-20
        # cov += npm.eye(cov.shape[0], cov.shape[1])
    return float(w / (((2 * np.pi) ** (len(mu) / 2)) * det ** (1 / 2)))


def power_term(x, w, mu, cov):
    det = np.linalg.det(cov)
    if det < 10**-20:
        cov += npm.eye(cov.shape[0], cov.shape[1])*0.00001
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
    model = np.zeros(nom, dtype=[('w', float),
                                 ('init_mu', float, d),
                                 ('mu', float, d),
                                 ('init_cov', float, (d, d)),
                                 ('cov', float, (d, d))])
    model['w'] = 1/nom
    if data.shape[1] > nom:
        mu = kmeans.find_centers(data.T, nom)
        model['mu'] = np.array(mu)
        for i in range(nom):
            # model['cov'][i] = np.cov(np.array(clus[i]).T)
            # if np.isnan(model['cov'][i]).any():
            model['cov'][i] = npm.eye(d)
    else:
        for i in range(nom):
            model['mu'][i] = data[:, np.int(np.random.random()*data.shape[1])]
            model['cov'][i] = npm.eye(d)
    model['init_mu'] = model['mu']
    # model['cov'] = np.random.random((nom, d, d))
    # for i in range(nom):
    #     # model['mu'][i] = np.random.randint(0,5,np.size(data, 0))
    #     model['cov'] = npm.eye(d)
        # model['cov'][i] = np.cov(np.array(clus[i]).T)
        # while np.isnan(model['cov'][i]).any():
        #     model['cov'][i] = np.cov(np.array(clus[i]).T)
    model['init_cov'] = model['cov']
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

    model_update = model

    if plot == 1:
        plot_model(model)
        plt.scatter(data[0, :], data[1, :], s=5)

    power = np.zeros((N, nom))
    weight = np.zeros((N, nom))
    for k in range(runs):
        for i in range(N):
            for j in range(nom):
                power[i][j] = power_term(data[:, i], model['w'][j], model['mu'][j], model['cov'][j])
                weight[i][j] = weight_term(data[:, i], model['w'][j], model['mu'][j], model['cov'][j])
        norm_power = (power.T - np.amax(power, axis=1)).T
        est = np.multiply(np.exp(norm_power), weight)
        px = (est.T / np.sum(est, axis=1)).T
        model_update['w'] = np.sum(px,axis=0)/N

        for j in range(nom):
            model_update['mu'][j] = sum(px[i,j] * data[:,i] for i in range(N))/np.sum(px,axis = 0)[j]
            model_update['cov'][j] = sum(px[i,j] * np.outer(data[:,i] - model_update['mu'][j],data[:,i] - model_update['mu'][j]) for i in range(N))/np.sum(px,axis = 0)[j] + npm.eye(d,d)*0.0000001

        # for m in range(nom):
        #     det = np.linalg.det(model_update['cov'][m])
        #     if det == 0:
        #         print('zer'+m)
        #         model_update['cov'][m] = model['init_cov'][m]
        #         model_update['mu'][m] = model['init_mu'][m]
        model = model_update
        if plot == 1:
            plot_model(model)
            plt.scatter(data[0, :], data[1, :], s=5)
        if progress == 1:
            print((k/runs)*100)
    return model

# k = 1000
# j = 10
# data = np.zeros((2, k))
# for i in range(j):
#     data[:, i*np.int(k/j):(i+1)*np.int(k/j)] = np.random.random()*2 + np.random.random((2,np.int(k/j)))*np.random.random()
# #
# model = gmm_init(5, 2, data)
# # plot_model(model,data)
# start = timeit.default_timer()
# fitted = gmm_fit(data, model, 10, progress=1)
# stop = timeit.default_timer()
# print(stop - start)
# print(fitted)
# plot_model(fitted, data)
# plt.show()