import math
import json
from approximator import TwoVariableApproximator
from draw import Drawer


# samples = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.5],
#            [0.0, 1.0, 0.5], [1.0, 1.0, 1.0]]


def get_samples():
    s = []
    x1_range, x2_range = 10, 10
    for x1_ind in range(x1_range+1):
        for x2_ind in range(x2_range+1):
            x1 = (math.pi * x1_ind) / x1_range
            x2 = (math.pi * x2_ind) / x2_range
            y = math.sin(x1) + math.cos(x2)
            s.append([x1, x2, y])
    return s


refile = open("2varSamples.json", "r")
samples = json.loads(refile.read())
approximator = TwoVariableApproximator(samples, 3)
progress = [approximator.objective()]

for i in range(10000):
    approximator.one_gradient_descent_step()
    progress.append(approximator.objective())
    print(progress[-1])

obj_function_plot = Drawer(xlimits=(0, len(progress)), ylimits=(0, 1.2 * progress[0]))
for i in range(len(progress)):
    obj_function_plot.add_points([[i, progress[i]]])
print("bettas", approximator.bettas)
print("estimate:", approximator.approximation([0.5, 0.5]), approximator.approximation([0.75, 0.25]))
print("expected:", math.sin(0.5) + math.cos(0.5), math.sin(0.75) + math.cos(0.25))
obj_function_plot.draw()



