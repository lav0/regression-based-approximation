from approximator import TwoVariableApproximator


samples = [ [0.0, 0.0, 1.0], [1.0, 1.0, 0.0] ]

approximator = TwoVariableApproximator(samples, 3)
print(approximator.bettas)
print(approximator.approximation([2.0, 0.0]))
print(approximator.objective())