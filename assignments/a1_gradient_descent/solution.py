import numpy as np
from optalg.interface.nlp import NLP
from optalg.interface.objective_type import OT
import sys
sys.path.append("../..")


def solve(nlp: NLP):
    """
    Gradient descent with backtracking Line search


    Arguments:
    ---
        nlp: object of class NLP that only contains one feature of type OT.f.

    Returns:
        x: local optimal solution (1-D np.ndarray)


    Task:
    ---

    Implement a solver that does iterations of gradient descent
    with a backtracking line search

    x = x - k * Df(x),

    where Df(x) is the gradient of f(x)
    and the step size k is computed adaptively with backtracking line search 步长k是通过回溯线搜索自适应计算的。

    Notes:
    ---

    Get the starting point with:
    x = nlp.getInitializationSample()

    Use the following to query the problem:
    phi, J = nlp.evaluate(x)

    phi is a vector (1D np.ndarray); use phi[0] to access the cost value
    (a float number).

    J is a Jacobian matrix (2D np.ndarray). Use J[0] to access the gradient
    (1D np.array) of phi[0].

    """

    # sanity check on the input nlp
    assert len(nlp.getFeatureTypes()) == 1
    assert nlp.getFeatureTypes()[0] == OT.f

    # get start point
    x = nlp.getInitializationSample()
    step = 1
    decrease_ls = 0.01
    stepsize_decrement = 0.5
    stepsize_decrement_plus = 1.2
    sigma_max = 3
    tolerance = 0.001

    for i in range(1000):
        phi, J = nlp.evaluate(x)
        f_k = phi[0]
        gradient = J[0]
        d_k = -gradient
        x_k_plus_one = x + step*d_k
        phi_k_plus_one, J_k_plus_1 = nlp.evaluate(x_k_plus_one)
        f_k_plus_one = phi_k_plus_one[0]

        while ((f_k_plus_one - f_k) > decrease_ls*np.dot(gradient.T, (step*d_k)) and np.linalg.norm(step*d_k) >= tolerance):
            step = stepsize_decrement*step

        if(np.linalg.norm(step*d_k) < 0.0001):
            break

        x = x + step*d_k
        step = min(stepsize_decrement_plus*step, sigma_max)

    # max_it = 300
    # delta = 1e-4
    # ls_k = .0001
    # alpha_rate = .5
    # it_ls_max = 20
    # xprev = np.copy(x)
    # for i in range(max_it):
    #     phi, J = nlp.evaluate(x)
    #     fref = phi[0]
    #     D = J[0]
    #     D_norm_square = np.dot(D, D)
    #     alpha = 1. step
    #     found = False
    #     it_ls = 0
    #     while not found and it_ls < it_ls_max:
    #         x_l = x - alpha * D
    #         phi, _ = nlp.evaluate(x_l)
    #         f = phi[0]
    #         if (f < fref - ls_k * alpha * D_norm_square):
    #             found = True
    #             x = np.copy(x_l)
    #         it_ls += 1
    #         alpha *= alpha_rate
    #     if np.linalg.norm(x-xprev) < delta:
    #         break
    #     xprev = np.copy(x)

    return x
