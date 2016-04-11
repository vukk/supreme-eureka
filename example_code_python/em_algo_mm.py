# Aalto University, School of Science
# T-61.5140 Machine Learning: Advanced probabilistic Methods
# Author: antti.kangasraasio@aalto.fi, 2016

from numpy import outer, eye, ones, zeros, diag, log, sqrt, exp, pi
from numpy.linalg import inv, solve
from numpy.random import multivariate_normal as mvnormal, normal, gamma, beta, binomial
from scipy.special import gammaln

from em_algo import EM_algo

class EM_algo_MM(EM_algo):
    """
        A mixture of two linear models.
    """

    def reset(self):
        pass


    def draw(self, item):
        return None, None


    def logl(self):
        return None, None


    def EM_iter(self):
        pass

