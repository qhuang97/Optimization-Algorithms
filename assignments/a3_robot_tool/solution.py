import numpy as np
import sys
sys.path.append("../..")
from optalg.interface.nlp import NLP
import math


class RobotTool(NLP):
    """
    """

    def __init__(self, q0: np.ndarray, pr: np.ndarray, l: float):
        """
        Arguments
        ----
        q0: 1-D np.array
        pr: 1-D np.array
        l: float
        """
        self.q0 = q0
        self.pr = pr
        self.l = l

    def evaluate(self, x):
        """
        """
        # print('x:{}'.format(x))
        # x_k = np.pi/180 * x
        # x_k = x
        # print('x_k:{}'.format(x_k))
        # print('self.q0:{}'.format(self.q0))
        # print('self.pr:{}'.format(self.pr))
        # print('self.l:{}'.format(self.l))
        p1 = np.cos(x[0]) + (1/2) * np.cos(x[0] + x[1]) + (1/3 + x[3]) * np.cos(x[0] + x[1] + x[2])
        p2 = np.sin(x[0]) + (1/2) * np.sin(x[0] + x[1]) + (1/3 + x[3]) * np.sin(x[0] + x[1] + x[2])
  
        # r_p = p1 - self.pr
        # r_q = x - self.q0
      
        y = [p1-self.pr[0],
             p2-self.pr[1],
             np.sqrt(self.l)*(x[0]-self.q0[0]),
             np.sqrt(self.l)*(x[1]-self.q0[1]),
             np.sqrt(self.l)*(x[2]-self.q0[2]),
             np.sqrt(self.l)*(x[3]-self.q0[3])]
        
        print('y:{}'.format(y))
        
        J = [[-p2, -1/2*np.sin(x[0]+x[1])-1 / 3 * np.sin(x.sum()), -1/3*np.sin(x.sum())],
             [p1, 1/2*np.cos(x[0]+x[1]) + 1/3*np.cos(x.sum()), 1/3*np.cos(x.sum())],
             [np.sqrt(self.l),0,0,0],
             [0,np.sqrt(self.l),0,0],
             [0,0,np.sqrt(self.l),0],
             [0,0,0,np.sqrt(self.l)]]
        
        print('J:{}'.format(J))

        return np.array(y), np.array(J)

    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        # return the input dimensionality of the problem (size of x)
        n = len(self.q0)
        return n

    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        return self.q0
