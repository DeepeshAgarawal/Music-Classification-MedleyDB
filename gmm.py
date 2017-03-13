import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.mlab as mlab 
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'


def plot_model(model):
    delta = 0.025
    x = np.arange(0.0, 1.0, delta)
    y = np.arange(0.0, 1.0, delta)
    X, Y = np.meshgrid(x, y)
    z = np.zeros((np.size(x), np.size(y)))
    # sum of Gaussian
    for i in range(np.size(model)):
        ztemp = mlab.bivariate_normal(X, Y, np.sqrt(model['cov'][i][0, 0]), np.sqrt(model['cov'][i][1, 1]), model['mu'][i][0], model['mu'][i][1], model['cov'][i][0,1])
        z = np.add(z, ztemp)
    plt.figure()
    CS = plt.contour(X, Y, z)


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


def gmm_init(nom, d):
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
    model['mu'] = np.random.random((nom, d))
    model['cov'] = np.random.random((nom, d, d))
    for i in range(nom):
        model['cov'][i] = (model['cov'][i] + model['cov'][i].T)/2
    return model


def gmm_fit(data, model, runs=50):
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


    for k in range(runs):
        for j in range(nom):
            sumest = 0
            munum = np.zeros((1, d))
            covnum = np.zeros((d, d))
            for i in range(N):
                est = pdf_multivariate_gauss(data[:,i], model['w'][j], model['mu'][j], model['cov'][j])
                sumest = np.add(sumest, est)
                munum  = np.add(munum, est * data[:,i])
                covnum = np.add(covnum, est * np.outer(data[:,i] - model['mu'][j], data[:,i] - model['mu'][j]))
            model_update['w'][j] = sumest/N
            model_update['mu'][j] = munum/sumest
            model_update['cov'][j] = covnum/sumest
        model = model_update

    return model

data = np.random.random((2,200))
model = gmm_init(3,2)
fitted = gmm_fit(data, model, 10)

print ("1")