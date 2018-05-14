import math
import random
from draw import Drawer
from approximator import OneVariableApproximator


def get_samples():
    global num_of_samples
    loc_n = num_of_samples
    s = []
    for i in range(0, loc_n):
        x_sample = i * math.pi / loc_n
        y_sample = math.cos(x_sample)
        if random.uniform(0.0, 1.0) > 0.45:
            s.append([x_sample, y_sample])
    return s


def get_approximations(a_approximator):
    global num_of_samples
    s = []
    for i in range(0, num_of_samples):
        x_sample = i * 2 * math.pi / num_of_samples
        y_sample = a_approximator.approximation(x_sample)
        s.append([x_sample, y_sample])
    return s


samples = get_samples()

d = Drawer()
d.add_points(points=samples)

approximator = OneVariableApproximator(samples, 4)
approximator.shrank_learning_rate(index=3, factor=0.1)

approximator.one_gradient_descent_step()
progress = [approximator.objective()]
for i in range(1000):
    approximator.one_gradient_descent_step()
    progress.append(approximator.objective())
    if progress[-2] < progress[-1]:
        approximator.shrank_learning_rate(factor=0.95)

previous = progress[0]
for p in progress[1:]:
    print(p, previous - p)
    previous = p


d.add_points(points=get_approximations(approximator), c='b')
d.draw()
