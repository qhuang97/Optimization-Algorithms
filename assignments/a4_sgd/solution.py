import sys
sys.path.append("../..")
from optalg.interface.nlp_stochastic import NLP_stochastic
import numpy as np


def solve(nlp: NLP_stochastic):
    """
    stochastic gradient descent


    Arguments:
    ---
        nlp: object of class NLP_stochastic that contains one feature of type OT.f.

    Returns:
    ---
        x: local optimal solution (1-D np.ndarray)

    Task:
    ---
    See the coding assignment PDF in ISIS.

    Notes:
    ---

    Get the starting point with:
    x = nlp.getInitializationSample()

    Get the number of samples with:
    N = nlp.getNumSamples()

    You can query the problem with any index i=0,...,N (N not included)

    y, J = nlp.evaluate_i(x, i)

    As usual, get the cost function (scalar) and gradient (1-D np.array) with y[0] and J[0]

    The output (y,J) is different for different values of i and x.

    The expected value (over i) of y,J at a given x is SUM_i [ nlp.evaluate_i(x, i) ]  / N

    """

    x = nlp.getInitializationSample()
    N = nlp.getNumSamples()

    #
    # Write your code Here
    #
    max_interation = 100
    x_old = np.copy(x)
    for k in range(1,max_interation):
        N = nlp.getNumSamples()
        for i in range(0,N):
            y, J = nlp.evaluate_i(x, i)
            y_ini = y[0]
            J_ini = J[0] 
            
            found = False
            j = 0
            alpha = 0.1
            for lamda in range(1,100):
                # print('alpha:{}'.format(alpha))
                # print('l:{}'.format(l))

                # delta = - 2 * np.dot(J_ini,y_ini)
                delta = - J_ini
                step = alpha/(1+alpha*lamda*k)
                # print('step:{}'.format(step))

                while not found and j < 5:
                    # newton step
                    x_i = x + step * delta
                    y, J = nlp.evaluate_i(x_i, i)
                    y_k = y[0]

                    if (y_k - y_ini < (0.01 * np.dot(J_ini, step * delta))):
                        found = True
                        x = np.copy(x_i)
                        # print('alpha:{}'.format(alpha))
                        # print('lamda:{}'.format(lamda))
                        
                    j += 1
                
                if np.linalg.norm(x-x_old) < 0.05:
                    break

                x_old = np.copy(x)

    return x
