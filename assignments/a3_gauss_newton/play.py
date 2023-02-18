import sys
sys.path.append("../..")
import numpy as np

from solution import solve
from optalg.example_nlps.logistic import Logistic
from optalg.example_nlps.trajectory import PointTrajOpt
from optalg.example_nlps.f_r import F_R
from optalg.example_nlps.linear_least_squares import LinearLeastSquares
from optalg.interface.nlp_traced import NLPTraced


problem = Logistic()
x = solve(problem)
print(x)
print(problem.xopt)

# N = 40
# problem = NLPTraced(PointTrajOpt(N), 29 * 20)
# x = solve(problem)

# solution = np.array([0.07145219, 0.0047813, 0.14290438, 0.0095626, 0.21435657, 0.0143439,
#                              0.28580877, 0.01912519, 0.35726096, 0.02390649, 0.42871315, 0.02868779,
#                              0.50016534, 0.03346909, 0.57161753, 0.03825039, 0.64306972, 0.04303168,
#                              0.71452191, 0.04781298, 0.7859741, 0.05259428, 0.8574263, 0.05737558,
#                              0.92887849, 0.06215687, 1.00033068, 0.06693817, 1.07211503, 0.13895804,
#                              1.14389939, 0.2109779, 1.21568374, 0.28299777, 1.28746809, 0.35501763,
#                              1.35925245, 0.4270375, 1.4310368, 0.49905736, 1.50282116, 0.57107723,
#                              1.57460551, 0.64309709, 1.64638987, 0.71511696, 1.71817422, 0.78713682,
#                              1.78995858, 0.85915669, 1.86174293, 0.93117655, 1.93352729, 1.00319642,
#                              1.9385441, 1.07842688, 1.9435609, 1.15365734, 1.94857771, 1.22888779,
#                              1.95359452, 1.30411825, 1.95861133, 1.37934871, 1.96362814, 1.45457917,
#                              1.96864495, 1.52980963, 1.97366175, 1.60504009, 1.97867856, 1.68027055,
#                              1.98369537, 1.75550101, 1.98871218, 1.83073147, 1.99372899, 1.90596193,
#                              1.9987458, 1.98119239])

# print('x:{}'.format(x))
# print('x solution:{}'.format(solution))