import math
import numpy as np

from ..interface.nlp import NLP
from ..interface.objective_type import OT


class Hole(NLP):
    """
    f =  x^T C x  / ( a*a + x^T C x )
    sos = []
    eq = []
    ineq = []
    bounds = ( [ -inf , -inf, ... ], [ inf, inf, ...] )
    """

    def __init__(self, C, a):
        """
        C: np.array 2d
        a: float
        """
        assert(C.shape[0] == C.shape[1])
        self.C = C
        self.a = a
        self.n = C.shape[0]

    def evaluate(self, x):
        """
        See Also
        ------
        NLP.evaluate
        """
        xCx = x @ self.C @ x

        a2xCx = self.a * self.a + xCx
        f = xCx / a2xCx
        J = (2 * self.C @ x * self.a * self.a) / (a2xCx) ** 2
        return np.array([f]), J.reshape(1, -1)

    def getDimension(self):
        """
        See Also
        ------
        NLP.getDimension
        """
        return self.n

    def getFHessian(self, x):
        """
        See Also
        ------
        NLP.getFHessian
        """
        a = self.a
        C = self.C
        ddf = (-8 * a * a * (C@x)[None].T @ (C@x)
               [None]) / ((a * a + x @ C @ x) ** 3)
        ddf += 2 * a * a * C / ((a * a + x @ C @ x) ** 2)
        return ddf

    def getFeatureTypes(self):
        """
        See Also
        ------
        NLP.getFeatureTypes
        """
        return [OT.f]

    def getInitializationSample(self):
        """
        See Also
        ------
        NLP.getInitializationSample
        """
        return np.ones(self.n)

    def report(self, verbose):
        """
        See Also
        ------
        NLP.report
        """
        strOut = "Hole function C"
        return strOut
