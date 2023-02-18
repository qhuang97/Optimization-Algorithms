import numpy as np
from optalg.interface.nlp import NLP
from optalg.interface.objective_type import OT
import sys
sys.path.append("../..")


def solve(nlp: NLP):
    """
    solver for constrained optimization


    Arguments:
    ---
        nlp: object of class NLP that contains features of type OT.f, OT.r, OT.eq, OT.ineq

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

    To know which type (normal cost, least squares, equalities or inequalities) are the entries in
    the feature vector, use:

    types = nlp.getFeatureTypes()

    id_f = [i for i, t in enumerate(types) if t == OT.f]
    id_r = [i for i, t in enumerate(types) if t == OT.r]
    id_ineq = [i for i, t in enumerate(types) if t == OT.ineq]
    id_eq = [i for i, t in enumerate(types) if t == OT.eq]

    Note that getFHessian(x) only returns the Hessian of the term of type OT.f.

    For example, you can get the value and Jacobian of the equality constraints with y[id_eq] (1-D np.array), and J[id_eq] (2-D np.array).

    All input NLPs contain one feature of type OT.f (len(id_f) is 1). In some problems,
    there no equality constraints, inequality constraints or residual terms.
    In those cases, some of the lists of indexes will be empty (e.g. id_eq = [] if there are not equality constraints).

    """

    x = nlp.getInitializationSample()

    types = nlp.getFeatureTypes()
    id_f = [i for i, t in enumerate(types) if t == OT.f]
    id_r = [i for i, t in enumerate(types) if t == OT.r]
    id_ineq = [i for i, t in enumerate(types) if t == OT.ineq]
    id_eq = [i for i, t in enumerate(types) if t == OT.eq]

    #
    # Write your code Here
    #
   
    x_old = np.copy(x)
    mu = nu = 10  # penalty parameter
    lamda = np.zeros([len(id_ineq),])  # Lagrange multipliers
    y, J = nlp.evaluate(x)
    kappa = np.zeros([len(y[id_eq]),])
    while True:
        y, J = nlp.evaluate(x)
        f_ini = y[id_f[0]]  # cost of f
        r_ini = y[id_r]  # cost of r
        g_ini = y[id_ineq]

        h_ini = np.asarray(y[id_eq])
        J_h_ini = np.asarray(J[id_eq])
        
        J_g = np.array(J[id_ineq])
        # print('lamda:{}'.format(lamda))
        g_ini_square = 0
        D_g_test = np.zeros((len(J[id_f[0]]),))
        for i in range(len(id_ineq)):
            if (g_ini[i] >= 0 or lamda[i] > 0):
                g_ini_square += (mu * g_ini[i] * g_ini[i])
            else:
                (J_g.T)[:,i] = 0

                
        D_f = J[id_f[0]]  # J of f
        D_r = 2 * J[id_r].T.dot(r_ini)  # J of r
        D_g = 2 * J_g.T.dot(g_ini)  # J of g
        D_h = 2 * J_h_ini.T.dot(h_ini)  # J of h
          
        H_f = nlp.getFHessian(x)  # H only of f
        H_r = 2 * np.dot(J[id_r].T, J[id_r])  # H of r
        H_g = 2 * np.dot(J_g.T, J_g)  # penalty term H of g
        H_h = 2 * np.dot(J[id_eq].T, J[id_eq])  # penalty term H of h
        
        S_ini_ges =  f_ini + np.dot(r_ini, r_ini.T) + np.dot(lamda, g_ini.T) + g_ini_square + np.dot(kappa, h_ini.T) + nu * np.dot(h_ini, h_ini.T)
    
        D_ges = D_f + D_r + np.dot(lamda, J[id_ineq]) + mu * D_g + np.dot(kappa, J_h_ini) +  nu * D_h 
        H_ges = H_f + H_r + mu * H_g + nu * H_h
     
        found = False
        i = 0
        step = 1.0  # step
        # mu = nu = 10
        
        if np.min(np.linalg.eigvals(H_ges) > 0):
            delta =  - np.dot(np.linalg.inv(H_ges), D_ges.T)
        else:
            delta = - D_ges

        # print('delta:{}'.format(delta))

        while not found:
            # newton step
            x_k = x + step * delta
            print('x_k:{}'.format(x_k))
        
            phi, _ = nlp.evaluate(x_k)
            f_k = phi[id_f[0]]
            r_k = phi[id_r]
            g_k = phi[id_ineq]
            h_k = phi[id_eq]

            g_k_square = 0
            for i in range(len(g_k)):
                if (g_k[i] >= 0 or lamda[i] > 0):
                    g_k_square += (mu * g_k[i] * g_k[i])
                    # g_k[i] = 0

            S_k_ges =  f_k + np.dot(r_k, r_k.T) + np.dot(lamda, g_k.T) + g_k_square + nu * np.dot(h_k, h_k.T) + np.dot(kappa, h_k.T)
    
            if (S_k_ges - S_ini_ges < (0.01 * np.dot(D_ges, step * delta))) :
                found = True
                x = np.copy(x_k)

            step *= 0.0005
            lamda = lamda + 2 * mu * g_k
            lamda[lamda < 0] = 0
            kappa = kappa + 2 * nu * h_k
            mu = 1.2 * mu
            nu = 1.2 * nu

        if np.linalg.norm(x-x_old) < 0.001:
            break

        x_old = np.copy(x)

    return x
