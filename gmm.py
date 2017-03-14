import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import timeit

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'


def plot_model(model, data):
    delta = 0.025
    x = np.arange(0.0, 1.0, delta)
    y = np.arange(0.0, 1.0, delta)
    X, Y = np.meshgrid(x, y)
    z = np.zeros((np.size(x), np.size(y)))
    # sum of Gaussian
    plt.figure()
    for i in range(np.size(model)):
        ztemp = mlab.bivariate_normal(X, Y, np.sqrt(model['cov'][i][0, 0]), np.sqrt(model['cov'][i][1, 1]), model['mu'][i][0], model['mu'][i][1], model['cov'][i][0,1])
        CS = plt.contour(X, Y, model['w'][i]*ztemp)
        z = np.add(z, model['w'][i] * ztemp)
    plt.plot(data[0, :], data[1, :], 'ro')
    plt.figure()
    CS = plt.contour(X, Y, z)
    plt.plot(data[0, :], data[1, :], 'ro')


def pdf_multivariate_gauss(x, w, mu, cov):
    """
    Calculate multivariate gauss density
        x = numpy array "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    """

    weight = 1 / (((2 * np.pi) ** (len(mu) / 2)) * (np.linalg.det(cov) ** (1 / 2)))
    power_term = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))
    return float(w * weight * np.exp(power_term))


def gmm_init(nom, d, data):
    """

    :param nom: number of Gaussian models
    :param d: dimension of the feature vector
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
        model['mu'][i] = data[:, np.int(np.random.random()*np.size(data, 1))]
        model['cov'][i] = npm.eye(d)
    return model


def gmm_fit(data, model, runs=50, plot = 0, progress = 0):
    """

    :param data: data points 'd x N' where 'd' number of features ; 'N' number of samples
    :param model: the structure which contains the gmm mean and cov ie x from gmm_init
    :param runs: number of times the model updates
    :return: model_updated 'runs' number of times
    """

    d = np.size(data,0)
    N = np.size(data, 1)
    nom = np.size(model)

    model_update = np.zeros(nom, dtype=[('w', float),
                           ('mu', float, d),
                           ('cov', float, (d, d))])
    model_update_2 = np.zeros(nom, dtype=[('w', float),
                                        ('mu', float, d),
                                        ('cov', float, (d, d))])
    if plot == 1:
        plot_model(model)
        plt.plot(data[0, :], data[1, :], 'ro')
    est=np.zeros(nom)
    for k in range(runs):
        sumpx = np.zeros(nom)
        munum = np.zeros((d, nom))
        covnum = np.zeros((d, d, nom))
        for i in range(N):
            for j in range(nom):
                est[j] = pdf_multivariate_gauss(data[:, i], model['w'][j], model['mu'][j], model['cov'][j])
            fx = np.sum(est)
            px = est/fx
            sumpx = np.add(sumpx, px)
            for l in range(nom):
                munum[:, l] = np.add(munum[:, l], data[:, i] * px[l])
                covnum[:, :, l] = np.add(covnum[:, :, l], px[l] * np.outer(data[:, i] - model['mu'][l], data[:, i] - model['mu'][l]))
        model_update['w'] = sumpx/N
        model_update['mu'] = (munum/sumpx).T
        model_update['cov'] = (covnum/sumpx).T
        # for l in range(nom):
        #     model_update_2['w'][l] = sumpx[l]/N
        #     model_update_2['mu'][l] = munum[:, l]/sumpx[l]
        #     model_update_2['cov'][l] = covnum[:, :, l]/sumpx[l]
        model = model_update
        if plot == 1:
            plot_model(model)
            plt.plot(data[0, :], data[1, :], 'ro')
        if progress == 1:
            print((k/runs)*100)
    return model

data = np.zeros((2, 200))
data[:, 0:75] = 0.1 + np.random.random((2,75))*0.4
data[:, 75:125] = np.random.random((2,50))
data[:, 125:200] = 0.5 + np.random.random((2,75))*0.1
model = gmm_init(3,2, data)
plot_model(model,data)
start = timeit.default_timer()
fitted = gmm_fit(data, model, 100,progress=1)
stop = timeit.default_timer()
print (stop - start)
plot_model(fitted, data)
plt.show()