import math
import numpy as np

from ..interface.nlp import NLP
from ..interface.objective_type import OT


class QuaterCircle(NLP):
    """
    x = [ x1 , x2 ] , x1, x2 \in \R
    f =  [x1+x2]
    sos = []
    eq = []
    ineq = [|x|**2-1, -x_1, -x_2]
    bounds = ( [ -inf , -inf], [ inf, inf] )
    """

    def __init__(self):
        """
        a: float
        b: float
        """
        pass

    def evaluate(self, x):
        """
        See Also
        ------
        MathematicalProgram.evaluate
        """
        x1 = x[0]
        x2 = x[1]

        phi = np.array([x1 + x2, x1**2 + x2**2 - 1, -x1, -x2])
        J = np.array([
            [1., 1.],
            [2 * x1, 2 * x2],
            [-1, 0],
            [0, -1]
        ])
        return phi, J

    def getFeatureTypes(self):
        """
        returns
        -----
        output: list of feature Types

        """
        return [OT.f, OT.ineq, OT.ineq, OT.ineq]

    def getDimension(self):
        """
        return the dimensionality of x

        Returns
        -----
        output: integer

        """
        return 2

    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        return np.array([.5, .5])
#         return np.array([.9, 0])

    def getFHessian(self, x):
        dim = self.getDimension()
        return np.zeros((dim, dim))
