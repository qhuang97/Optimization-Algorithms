import numpy as np
from optalg.interface.nlp import NLP
import sys
sys.path.append("../..")


class NLP_nonlinear(NLP):
    """
    Nonlinear program with cost  1 / || C x ||^2
    x in R^n
    C in R^(m x n)
    || . || is the 2-norm
    feature types: [ OT.f ]

    """

    def __init__(self, C):
        """
        """
        self.C = C

    def evaluate(self, x):
        """
        Returns the features and the Jacobians
        of a nonlinear program.
        In this case, we have a single feature (the cost function)
        because there are no constraints or residual terms.
        Therefore, the output should be:
            y: the feature (1-D np.ndarray of shape (1,)) 
            J: the jacobian (2-D np.ndarray of shape (1,n))

        See also:
        ----
        NLP.evaluate
        """

        y = np.array([1/(np.linalg.norm(np.dot(self.C, x)))**2])
        J = np.array([(-2 * self.C.T @ self.C @ x) / np.linalg.norm(np.dot(self.C, x))**4])
    

        return y, J

    def getDimension(self):
        """
        Returns the dimensionality of the variable x

        See Also
        ------
        NLP.getDimension
        """

        n = self.size
        return n

    def getFHessian(self, x):
        """
        Returns the hessian of the cost term.
        The output should be: 
            H: the hessian (2-D np.ndarray of shape (n,n))

        See Also
        ------
        NLP.getFHessian
        """
        # add code to compute the Hessian matrix
        H = np.array([(-2*self.C.T @ self.C * np.linalg.norm(np.dot(self.C, x))**2 + 8*(self.C.T @ self.C @ x)**2)/(np.linalg.norm(np.dot(self.C, x)))**6])

        return H

    def getInitializationSample(self):
        """
        See Also
        ------
        NLP.getInitializationSample
        """
        return np.ones(self.getDimension())

    def report(self, verbose):
        """
        See Also
        ------
        NLP.report
        """
        return "Nonlinear function  1 / || C x ||^2"
