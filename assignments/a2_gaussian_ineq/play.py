import sys
sys.path.append("../..")
import numpy as np

from solution import NLP_Gaussina_ineq
from optalg.interface.nlp import NLP
from optalg.utils.finite_diff import finite_diff_J, finite_diff_hess


# You can freely modify this script to play around with
# the implementation of your NLP


# Example:
# D = np.ones((2, 2))
# A = np.ones((2, 2))
# b = np.ones(2)
# x0 = np.zeros(2)

# problem = NLP_Gaussina_ineq(x0, D, A, b)
# x = np.ones(2)
# y, J = problem.evaluate(x)
# H = problem.getFHessian(x)

D = np.array([[2, 1], [1, 2]])
A = np.array([[1., 2.], [3., 4.]])
b = np.array([0., 1.])
x0 = np.array([.5, .1])

problem = NLP_Gaussina_ineq(x0, D, A, b)

x = np.array([-1, .1])
H = problem.getFHessian(x)

def f(x):
    return problem.evaluate(x)[0][0]

tol = 1e-4
Hdiff = finite_diff_hess(f, x, tol)
print('H:{}'.format(H))
print('Hdiff:{}'.format(Hdiff))
