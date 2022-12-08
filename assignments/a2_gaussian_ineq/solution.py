import numpy as np
from optalg.interface.objective_type import OT
from optalg.interface.nlp import NLP
import sys
sys.path.append("../..")


class NLP_Gaussina_ineq(NLP):
    """
    Nonlinear program

    Cost:  - exp ( - (x - x0)^T D (x - x0))
    Inequalities: Ax <= b
    Variable: x in R^n

    Parameters:
    x0 in R^n
    D in R^(nxn) symmetric
    A in R^(mxn)
    b in R^m

    ^T means transpose
    exp is exponential function

    Feature types: [ OT.f ] +  m * [ OT.ineq ]

    """

    def __init__(self, x0, D, A, b):
        """
        """
        self.x0 = x0
        self.D = D
        self.A = A
        self.b = b

    def evaluate(self, x):
        """
        Returns the features (y) and the Jacobian (J) of the nonlinear program.

        In this case, we have 1 cost function and m inequalities.
        The cost should be in the first entry (index 0) of the feature
        vector. The inequality features should come next, following the
        natural order in Ax<=b. That is, the first inequality (second entry of
        the feature vector) is A[0,:] x <= b[0], the second inequality
        is A[1,:] x <= b[1] and so on.

        The inequality entries should be written in the form y[i] <= 0.
        For example, for inequality x[0] <= 1 --> we use feature
        y[i] = x[0] - 1.

        The row i of the Jacobian J is the gradient of the entry i in
        the feature vector, e.g. J[0,:] is the gradient of y[0].

        Therefore, the output should be:
            y: the feature (1-D np.ndarray of shape (1+m,))
            J: the Jacobian (2-D np.ndarray of shape (1+m,n))

        See also:
        ----
        NLP.evaluate
        """
        x_l = x - self.x0
        m = len(self.b)
        n = len(x)
        y = np.zeros((m+1,)).astype(float)

        J = np.zeros((m+1, n)).astype(float)
        y0 = -np.exp(-x_l.T @ self.D @ x_l)  # 1,1

        g = np.dot(self.A, x) - self.b

        t1 = self.D @ x_l
        J0 = 2 * np.exp(-x_l.T @ t1) * t1

        for i in range(0, m+1):
            if(i == 0):
                y[i] = y0
            else:
                y[i] = g[i-1]

        for i in range(0, 1+m):
            for j in range(0, n):
                if(i == 0):
                    J[i, j] = J0[j]
                else:
                    J[i, j] = self.A[i-1, j]

        return y, J

    def getDimension(self):
        """
        Return the dimensionality of the variable x

        See Also
        ------
        NLP.getDimension
        """

        n = len(self.x0)
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

        x_l = x - self.x0  # n,1
        t1 = self.D.dot(x_l)   # n,1
        t2 = np.exp(-x_l.dot(t1))  # 1,1
        H = 2 * t2 * self.D - 4 * t2 * np.outer(t1, x_l.dot(self.D))
        return H

    def getInitializationSample(self):
        """
        See Also
        ------
        NLP.getInitializationSample
        """
        return np.ones(self.getDimension())

    def getFeatureTypes(self):
        """
        See Also
        ------
        NLP.getInitializationSample
        """
        return [OT.f] + self.A.shape[0] * [OT.ineq]

    def report(self, verbose):
        """
        See Also
        ------
        NLP.report
        """
        return "Gaussian function with inequalities"
