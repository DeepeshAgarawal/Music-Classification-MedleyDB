import numpy as np


def pdf_multivariate_gauss(x, mu, cov):
    """
    Calculate multivariate gauss density
        x = numpy array "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    """

    weight = 1 / (((2 * np.pi) ** (len(mu) / 2)) * (np.linalg.det(cov) ** (1 / 2)))
    power_term = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))
    return float(weight * np.exp(power_term))



