class OneVariableApproximator:
    def __init__(self, training_data, degree):
        self.training_data = training_data
        self.degree = degree
        # the coefficients of polynomial
        self.bettas = [0.0] * self.degree
        # learning rate
        self.alfas = [0.1 / (x+1) for x in range(self.degree)]

    def shrank_learning_rate(self, index=-1, factor=0.9):
        if index == -1:
            for i in range(self.degree):
                self.alfas *= factor
        else:
            self.alfas[index] *= factor

    def approximation(self, single_input):
        x = single_input
        result = 0
        for b, i in zip(self.bettas, range(self.degree)):
            result += b * x**i
        return result

    def objective(self):
        n = len(self.training_data)
        sum_of_squares = 0
        for x, y in self.training_data:
            y_hat = self.approximation(x)
            sum_of_squares += (y_hat - y) ** 2
        sum_of_squares *= 1.0 / (2 * n)
        return sum_of_squares

    def gradient(self):
        grad = [0] * self.degree
        for x, y in self.training_data:
            y_hat = self.approximation(x)
            s = (y_hat - y)
            for i in range(self.degree):
                grad[i] += self.alfas[i] * s * x**i
            # grad[0] += learning_rate[0] * s * 1
            # grad[1] += learning_rate[1] * s * x
            # grad[2] += learning_rate[2] * s * x ** 2
            # grad[3] += learning_rate[3] * s * x ** 3
        return [component * (1.0 / len(self.training_data)) for component in grad]

    def one_gradient_descent_step(self):
        grad = self.gradient()
        for index in range(self.degree):
            self.bettas[index] -= grad[index]


