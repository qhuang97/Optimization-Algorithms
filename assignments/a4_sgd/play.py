import sys
sys.path.append("../..")
from optalg.interface.objective_type import OT
from optalg.interface.nlp import NLP
import numpy as np
from optalg.example_nlps.quadratic_stochastic import Quadratic_stochastic
from optalg.example_nlps.linear_least_squares_stochastic import Linear_least_squares_stochastic
from solution import solve
import matplotlib.pyplot as plt


# A = np.array([[0.85880983, 0.42075752, 0.14625862],
#               [0.57705246, 0.76635021, 0.64077446],
#               [0.85916002, 0.86186594, 0.81231712],
#               [1., 0., 0.],
#               [0., 1., 0.],
#               [0., 0., 1.]])

# b = np.zeros(6)

# problem = Linear_least_squares_stochastic(A, b)

Qs = [np.diag([1., 1, 1]), np.diag([2., 1, 4]), np.diag(
            [10., 0, 4]), np.diag([0., 1, 1]), np.diag([0., 0, 0.1])]

gs = [np.array([0.13359639,
                0.20692249,
                0.4099303]),
    np.array([0.86686032,
                0.26361185,
                0.48886114]),
    np.array([0.01843443,
                0.89926551,
                0.3261251]),
    np.array([0.86751661,
                0.95695649,
                0.68315632]),
    np.array([0.90034537,
                0.96245117,
                0.84465686])]

problem = Quadratic_stochastic(Qs, gs)

x = solve(problem)

# plt.plot(x)
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
# plt.show()
print("output", x)
print("otpimal solution", problem.opt)
