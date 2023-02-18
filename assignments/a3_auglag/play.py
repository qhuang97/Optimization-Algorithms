import sys
sys.path.append("../..")
import numpy as np
from optalg.example_nlps.linear_program_ineq import LinearProgramIneq
from optalg.example_nlps.quadratic_program import QuadraticProgram
from optalg.example_nlps.quarter_circle import QuaterCircle
from optalg.example_nlps.halfcircle import HalfCircle
from optalg.example_nlps.logistic_bounds import LogisticWithBounds
from optalg.example_nlps.nonlinearA import NonlinearA
from optalg.interface.nlp_traced import NLPTraced
from optalg.example_nlps.f_r_eq import F_R_Eq
from solution import *
from optalg.utils.finite_diff import *

# problem = LinearProgramIneq(2)
# x = solve(problem)
# solution = np.zeros(2)

FACTOR = 30

def _solve(nlp):
    return solve(nlp)

# problem = NLPTraced(LinearProgramIneq(2), max_evaluate=39 * FACTOR)
# x = _solve(problem)
# solution = np.zeros(2)
# print("x", x)
# print("solution", solution)



H = np.array([[1., -1.], [-1., 2.]])
g = np.array([-2., -6.])
Aineq = np.array([[1., 1.], [-1., 2.], [2., 1.]])
bineq = np.array([2., 2., 3.])
problem = NLPTraced(
            QuadraticProgram(
                H=H,
                g=g,
                Aineq=Aineq,
                bineq=bineq),
            max_evaluate=54 * FACTOR)

problem.getInitializationSample = lambda: np.zeros(2)
x = _solve(problem)
solution = np.array([0.6667, 1.3333])
print("x", x)
print("solution", solution)
