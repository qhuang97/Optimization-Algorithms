import numpy as np
from optalg.interface.nlp import NLP
from optalg.interface.objective_type import OT
import sys
sys.path.append("../..")


def solve(nlp: NLP):
    """
    solver for unconstrained optimization, including least squares terms

    Arguments:
    ---
        nlp: object of class NLP that contains one feature of type OT.f,
            and m features of type OT.r

    Returns:
        x: local optimal solution (1-D np.ndarray)


    Task:
    ---
    See the coding assignment PDF in ISIS.


    Notes:
    ---

    Get the starting point with:

    x = nlp.getInitializationSample()

    You can query the problem with:

    y,J = nlp.evaluate(x)
    H = npl.getFHessian(x)

    To know which type (normal cost or least squares) are the entries in
    the feature vector, use:

    types = nlp.getFeatureTypes()

    id_f = [i for i, t in enumerate(types) if t == OT.f]
    id_r = [i for i, t in enumerate(types) if t == OT.r]

    The total cost is:

    y[self.id_f[0]] + np.dot(y[self.id_r], y[self.id_r])

    Note that getFHessian(x) only returns the Hessian of the term of type OT.f.

    For example, you can get the value and Jacobian of the least squares terms with y[id_r] (1-D np.array), and J[id_r] (2-D np.array).

    The input NLP contains one feature of type OT.f (len(id_f) is 1) (but sometimes f = 0 for all x).
    If there are no least squares terms, the lists of indexes id_r will be empty (e.g. id_r = []).

    """
    x = nlp.getInitializationSample()

    types = nlp.getFeatureTypes()

    id_f = [i for i, t in enumerate(types) if t == OT.f]
    id_r = [i for i, t in enumerate(types) if t == OT.r]

    #
    # Write your code Here
    #
    x_old = np.copy(x)

    # for i in range(1000):
    while True:
        y, J = nlp.evaluate(x)

        f_ini = y[id_f[0]]  # cost of f
        r_ini = y[id_r]  # cost of r

        D_f = J[id_f[0]]  # J of f
        D_r = 2 * J[id_r].T.dot(r_ini)  # J of r

        H_f = nlp.getFHessian(x)  # H only of f
        H_r = 2 * np.dot(J[id_r].T, J[id_r]) # H of r

        G_ini_ges = f_ini + np.dot(r_ini.T, r_ini)
        D_ges = D_r + D_f
        H_ges = H_r + H_f

        print('f_ini:{}'.format(f_ini))
        print('r_ini:{}'.format(r_ini))
        # print('J_f:{}'.format(J[id_f[0]]))
        print('J_r:{}'.format(J[id_r]))
        print('D_f:{}'.format(D_f))
        print('D_r:{}'.format(D_r))
        print('D_ges:{}'.format(D_ges))
        print('G_ini_ges:{}'.format(G_ini_ges))
        print('H_f:{}'.format(H_f))
        print('H_r:{}'.format(H_r))
        print('H_ges:{}'.format(H_ges))

        found = False
      
        step = 1.0  # step

        if np.min(np.linalg.eigvals(H_ges) > 0):
            delta = - np.dot(np.linalg.inv(H_ges), D_ges.T)
        else:
            delta = - D_ges
            
        print('delta:{}'.format(delta))
        while not found:
            # newton step
            x_k = x + step * delta
            print('x_k:{}'.format(x_k))

            phi, _ = nlp.evaluate(x_k)
            f_k = phi[id_f[0]]
            r_k = phi[id_r]

            G_k_ges = f_k + np.dot(r_k.T, r_k)

            if (G_k_ges - G_ini_ges < (0.01 * np.dot(D_ges, step * delta))):
                # if True:
                found = True
                x = np.copy(x_k)

            step *= 0.5

        if np.linalg.norm(x-x_old) < 0.001:
            break

        x_old = np.copy(x)

    return x
