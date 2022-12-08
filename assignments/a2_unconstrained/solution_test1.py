import numpy as np
from optalg.interface.nlp import NLP
from optalg.interface.objective_type import OT
import sys
sys.path.append("../..")


def solve(nlp: NLP):
    """
    Solver for unconstrained optimization


    Arguments:
    ---
        nlp: object of class NLP that only contains one feature of type OT.f.

    Returns:
        x: local optimal solution (1-D np.ndarray)


    Task:
    ---

    See instructions and requirements in the coding assignment PDF in ISIS.

    Notes:
    ---

    Get the starting point with:
    x = nlp.getInitializationSample()

    Use the following to query the function and gradient of the problem:
    phi, J = nlp.evaluate(x)

    phi is a vector (1D np.ndarray); use phi[0] to access the cost value
    (a float number).

    J is a Jacobian matrix (2D np.ndarray). Use J[0] to access the gradient
    (1D np.array) of phi[0].

    Use getFHessian to query the Hessian.

    H = nlp.getFHessian(x)

    H is a matrix (2D np.ndarray) of shape n x n.


    """

    # sanity check on the input nlp
    assert len(nlp.getFeatureTypes()) == 1
    assert nlp.getFeatureTypes()[0] == OT.f

    # get start point
    x = nlp.getInitializationSample()
    #
    # Write your code here
    thet = 1e-4
    rho_ls = 0.01
    step_rate = 0.5
    it_ls_max = 20
    step = 1.0  # step

    x_old = np.copy(x)

    # lambda1 = 0.01

    for i in range(100):
        phi, J = nlp.evaluate(x)
        H = nlp.getFHessian(x)
        f_ini = phi[0]  # cost
        D = J[0]  # gradient
        found = False
        it_ls = 0
        # step = 1.0  # step

        # delta = np.dot(np.linalg.inv(H), D)  # step direction
        delta = (D * (1 / np.linalg.norm(D)))  # step direction

        if(np.all((D.T @ delta) > 0)):
            # if(np.all(H > 0)):
            # delta = np.dot(D, np.linalg.inv(H))
            delta = - (D * (1 / np.linalg.norm(D)))
        else:
            # delta = np.dot(D, np.linalg.inv(H))
            delta = D * (1 / np.linalg.norm(D))

        # print('D{}'.format(D))
        # print('H{}'.format(H))

        # if lambda1 > min(np.linalg.eigvals(H)):
        #     print(min(np.linalg.eigvals(H)))
        #     delta = - np.dot(D, np.linalg.inv(H))
        #     delta = - D * (1 / np.linalg.det(D))

        while not found and it_ls < it_ls_max:
            # print('delta{}'.format(delta))

            # newton step
            x_k = x + step * delta
            phi, _ = nlp.evaluate(x_k)
            f_k = phi[0]
            if (f_k - f_ini <  rho_ls * np.dot(D, step * delta)) :
                found = True
                x = np.copy(x_k)
                # print('x{}'.format(x))

            it_ls += 1
            step *= step_rate
            step = min(1.2 * step, 1)
            # x = x + step * delta
            # step = min(1.2 * step, 1)
        # precision require
        # print(step*delta)
        # xl = x-x_old
        # print('xl{}'.format(xl))
        # if np.linalg.norm(step*delta) < 0.00000001:
        #     break
            
        if np.linalg.norm(x-x_old) < 0.000001 :
            break

        x_old = np.copy(x)
        # print('x_old{}'.format(x_old))

    return x
