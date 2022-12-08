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
        step = 1.0  # step

        delta = - np.dot(np.linalg.inv(H), D)  # step direction

        if(np.all((D.T @ delta) > 0)):
            delta = - D
        else:
            delta = - np.dot(np.linalg.inv(H), D)

        while not found and it_ls < it_ls_max:
            # newton step
            x_k = x + step * delta
            phi, _ = nlp.evaluate(x_k)
            f_k = phi[0]
            if (f_k - f_ini <  rho_ls * np.dot(D, step * delta)) :
                found = True
                x = np.copy(x_k)

            it_ls += 1
            step *= step_rate
            # step = min(1.2 * step, 1)

        if np.linalg.norm(x-x_old) < 0.0001 :
            break

        x_old = np.copy(x)
      

    # get start point
    # step = 1
    # x_old = np.copy(x)
    # decrease_ls = 0.01
    # stepsize_decrement = 0.5
    # stepsize_decrement_plus = 1.2
    # sigma_max = 3
    # tolerance = 0.001

    # for i in range(500):
    #     phi, J = nlp.evaluate(x)
    #     f_k = phi[0]
    #     D = J[0]  # gradient
    #     H = nlp.getFHessian(x)
    #     d_k = - np.dot(np.linalg.inv(H), D)  # step direction
    #     # d_k = - D * (1 / np.linalg.norm(D))

    #     # d_k = D

    #     if(np.any((D.T @ d_k) > 0)):
    #         # d_k = - (D * (1 / np.linalg.norm(D)))
    #         d_k = - D
    #     else:
    #         d_k = - np.dot(np.linalg.inv(H), D)
            
    #     x_k_plus_one = x + step*d_k
    #     phi_k_plus_one, _ = nlp.evaluate(x_k_plus_one)
    #     f_k_plus_one = phi_k_plus_one[0]

    #     while ((f_k_plus_one - f_k) > 0.01*np.dot(D.T, (step*d_k)) and np.linalg.norm(step*d_k) > 0.001):
    #         # not found
    #         step = 0.5*step
            
    #     if(np.linalg.norm(step*d_k) < 0.0001) and np.linalg.norm(x-x_old) < 0.0001:
    #         # found
    #         break

    #     x_old = np.copy(x)
    #     x = x + step*d_k
    #     step = min(1.2*step, 1)
       

    return x
