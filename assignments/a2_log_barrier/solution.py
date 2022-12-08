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

    # Write your code Here
    x_old = np.copy(x)

    for i in range(1000):
        y, J = nlp.evaluate(x)
        f_ini = y[id_f[0]]
        D = J[id_f[0]]
        H = nlp.getFHessian(x)
        found = False
        iter = 0
        step = 1
        mu = 1
        print(D)
        print('H eigenvalue:{}'.format(np.linalg.eigvals(H)))

        # if (np.all(D < 0)):
        #     d_k = D
        # else:
        #     d_k = -D

        if np.min(np.linalg.eigvals(H) > 0):
            d_k = -np.dot(np.linalg.inv(H), D)
        else:
            d_k = -D

        print('dk: {}'.format(d_k))

        while not found and iter < 30:
            x_l = x + step * d_k
            y, _ = nlp.evaluate(x_l)

            f = y[id_f[0]]
            g = y[id_ineq]

            if(min(g) > 0):
                f = np.inf
            # else:
            #     B = f - mu*np.sum(np.log(-g))
            # x_star = np.argmin(B)
            # y_star, _ = nlp.evaluate(x_star)

            if min(g) <= 0 and ((f - f_ini) < 0.01*np.dot(D.T, (step*d_k))):
                found = True
                x = np.copy(x_l)

            # B = f - mu*np.sum(np.log(-g))
            # x_star = np.argmin(B)

            iter += 1
            mu *= 0.5
            step *= 0.5

        if np.linalg.norm(x-x_old) < 1e-4:
            break
        x_old = np.copy(x)

    return x
