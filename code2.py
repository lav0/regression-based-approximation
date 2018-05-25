from approximator import TwoVariableApproximator


samples = [ [0.0, 0.0, 1.0], [1.0, 1.0, 6.0] ]

approximator = TwoVariableApproximator(samples, 3)
for b in approximator.bettas:
    print(b)
grad = approximator.gradient()
for g in grad:
    print(g)
print(approximator.approximation([2.0, 0.0]))
print(approximator.objective())
