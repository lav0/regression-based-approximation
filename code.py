import math
import random
from draw import Drawer
from approximator import OneVariableApproximator

num_of_samples = 40


def get_samples():
    global num_of_samples
    loc_n = num_of_samples
    s = []
    for i in range(0, loc_n):
        if loc_n // 5 < i < 2 * loc_n // 5 or 3 * loc_n // 5 < i < 4 * loc_n // 5:
            continue
        x_sample = i * math.pi / loc_n
        y_sample = math.cos(x_sample)
        s.append([x_sample, y_sample])
    s.append([1.55 * math.pi, math.cos(1.55 * math.pi)])
    s.append([1.45 * math.pi, math.cos(1.45 * math.pi)])
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

main_plot = Drawer()
main_plot.add_points(points=samples)

approximator = OneVariableApproximator(samples, 5)

progress = [approximator.objective()]
for i in range(10000):
    approximator.one_gradient_descent_step()
    progress.append(approximator.objective())

obj_function_plot = Drawer(xlimits=(0, len(progress)), ylimits=(0, 1.2 * progress[0]))
for i, p in zip(range(len(progress)), progress):
    pair = [i, p]
    print(pair)
    obj_function_plot.add_points([pair])


main_plot.add_points(points=get_approximations(approximator), c='b')
main_plot.draw()
obj_function_plot.draw()
