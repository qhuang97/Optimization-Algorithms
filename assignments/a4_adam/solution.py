import sys
sys.path.append("../..")
from optalg.interface.nlp_stochastic import NLP_stochastic
import numpy as np


def solve(nlp: NLP_stochastic):
    """
    stochastic gradient descent -- ADAM


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
    max_interation = 2000
    x_old = np.copy(x)
    # step = 1e-2
    # lamda = 0.1
    rho1 = 0.9
    rho2 = 0.999
    S = 0
    M = 0
    for k in range(1,max_interation):
        N = nlp.getNumSamples()
        # i = np.random.randint(0,N)
        for i in range(0,N):
            y, J = nlp.evaluate_i(x, i)
            y_ini = y[0]
            J_ini = J[0]
            S = rho1*S + (1-rho1) * J_ini  # update 1. moment vector
            M = rho2*M + (1-rho2) * np.dot(J_ini,J_ini) # update2. moment vector
           
            S_cor = S/(1- pow(rho1,k)) #bias correct
            M_cor = M/(1- pow(rho2,k)) #bias correct

            print('S:{}'.format(S))
            print('M:{}'.format(M))

            found = False
            j = 0
            alpha = 0.001 #stepsize
            sigma = 1e-8
            step = -alpha/(np.sqrt(M_cor + sigma * np.ones(len(x),)))
            # step = -alpha/ (M_cor + sigma)
            # print('S_cor:{}'.format(S_cor))
        
            while not found and j < 10:
                # newton step
                x_k = x - np.dot(alpha/(np.sqrt(M_cor + sigma * np.ones(len(x),))), S_cor.T)
                # x_k = x - alpha * S_cor/ (M_cor + sigma)
                y, J = nlp.evaluate_i(x_k, i)
                y_k = y[0]
                
                if (y_k - y_ini < (0.01 * np.dot(J_ini, step * S_cor))) :
                    found = True
                    x = np.copy(x_k)
                j += 1

            if np.linalg.norm(x-x_old) < 0.05:
                break

            x_old = np.copy(x)

    return x
