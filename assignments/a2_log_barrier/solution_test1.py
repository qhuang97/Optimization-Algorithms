import numpy as np
from optalg.interface.nlp import NLP
from optalg.interface.objective_type import OT
import sys
sys.path.append("../..")


def solve(nlp: NLP):
    """
    solver for constrained optimization (cost term and inequalities)


    Arguments:
    ---
        nlp: object of class NLP that contains one feature of type OT.f,
            and m features of type OT.ineq.

    Returns:
        x: local optimal solution (1-D np.ndarray)


    Task:
    ---
    See the coding assignment PDF in ISIS.


    Notes:
    ---

    Get the starting point with:

    x = nlp.getInitializationSample()

    To know which type (cost term or inequalities) are the entries in
    the feature vector, use:
    types = nlp.getFeatureTypes()

    Index of cost term
    id_f = [ i for i,t in enumerate(types) if t == OT.f ]
    There is only one term of type OT.f ( len(id_f) == 1 )

    Index of inequality constraints:
    id_ineq = [ i for i,t in enumerate(types) if t == OT.ineq ]

    Get all features (cost and constraints) with:

    y,J = nlp.evaluate(x)
    H = npl.getFHessian(x)

    The value, gradient and Hessian of the cost are:

    y[id_f[0]] (scalar), J[id_f[0]], H

    The value and Jacobian of inequalities are:
    y[id_ineq] (1-D np.array), J[id_ineq]


    """

    types = nlp.getFeatureTypes()
    id_f = [i for i, t in enumerate(types) if t == OT.f]
    id_ineq = [i for i, t in enumerate(types) if t == OT.ineq]

    x = nlp.getInitializationSample()

    #
    # Write your code Here
    #
    xprev = np.copy(x)
    thet = 0.0001
    mu = 1
    alpha = 1
    for i in range(300):
        y, J = nlp.evaluate(x)
        D = J[id_f[0]]
        print(D)
        H = nlp.getFHessian(x)
        f_ini = y[id_f[0]]
        found = False
        j = 0
        while not found and j < 20:

        

            # delta = - np.dot(np.linalg.inv(H), D)
            delta = np.dot(D,D)
            # print(delta)
            x_l = x + alpha * delta
            # x = argmin()
            y, _ = nlp.evaluate(x_l)
            f = y[id_f[0]]
            B = f - mu*np.sum(np.log(-y[id_ineq]))
            x_l = np.argmin(B)
            if ((f - f_ini) < (len(x)/(x_l*mu))):
                found = True
                x = np.copy(x_l)
            j += 1
            mu *= 0.5
        if np.linalg.norm(x-xprev) <  0.0001:
            break
        xprev = np.copy(x)
    return x
