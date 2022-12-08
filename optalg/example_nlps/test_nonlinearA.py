import sys
sys.path.append("../..")

from optalg.example_nlps.nonlinearA import NonlinearA as Problem
from optalg.interface.nlp import NLP
from optalg.utils.finite_diff import *
import numpy as np
import unittest
import math


class testNonLinearA(unittest.TestCase):
    """
    test class Constrained0
    """
    problem = Problem

    def testConstructor(self):
        self.problem()

    def testOptimum(self):
        # n = 4
        x = np.array([1, 0])
        problem = self.problem()
        phi, J = problem.evaluate(x)
        print("optimum")
        print("phi ", phi)
        print("types ", problem.getFeatureTypes())

        x = np.array([3.5, 1.5])
        phi, J = problem.evaluate(x)
        print("init ")
        print("phi ", phi)
        print("types ", problem.getFeatureTypes())

        # self.assertTrue(np.allclose(phi, np.zeros(n + 1)))

    def testValue1(self):
        # n = 4
        problem = self.problem()
        x = np.zeros(4)
        phi, J = problem.evaluate(x)
        # self.assertTrue(np.allclose(phi, np.zeros(n + 1)))

    def testJacobian(self):
        problem = self.problem()
        x = np.array([1, -1])
        flag, _, _ = check_nlp(
            problem.evaluate, x, 1e-5, True)
        self.assertTrue(flag)

    def testHessain(self):
        problem = self.problem()
        x = np.array([.8, -.5])

        H = problem.getFHessian(x)

        # we know that in this implementation, the first term is the
        # OT.f cost
        def f(x):
            return problem.evaluate(x)[0][0]

        tol = 1e-4
        Hdiff = finite_diff_hess(f, x, tol)
        flag = np.allclose(H, Hdiff, 10 * tol, 10 * tol)
        self.assertTrue(flag)

# usage:
# print results in terminal
# python3 test.py
# store results in file
# python3 test.py out.log


if __name__ == "__main__":
    if len(sys.argv) == 2:
        log_file = sys.argv.pop()
        with open(log_file, "w") as f:
            runner = unittest.TextTestRunner(f, verbosity=2)
            unittest.main(testRunner=runner)
    else:
        unittest.main()
