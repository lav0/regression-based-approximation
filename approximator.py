class OneVariableApproximator:
    def __init__(self, training_data, degree):
        self.training_data = training_data
        self.degree = degree
        # the coefficients of polynomial
        self.bettas = [0.0] * self.degree
        # learning rate
        self.alfa = 0.001

    def shrank_learning_rate(self, factor=0.9):
        self.alfa *= factor

    def approximation(self, single_input):
        x = single_input
        result = 0
        for b, i in zip(self.bettas, range(self.degree)):
            result += b * x**i
        return result

    def objective(self):
        n = len(self.training_data) + 1
        sum_of_squares = 0
        for x, y in self.training_data:
            y_hat = self.approximation(x)
            sum_of_squares += (y_hat - y) ** 2
        sum_of_squares *= 1.0 / (2 * n)
        return sum_of_squares

    def gradient(self):
        n = len(self.training_data) + 1
        grad = [0] * self.degree
        for x, y in self.training_data:
            y_hat = self.approximation(x)
            s = (y_hat - y)
            for i in range(self.degree):
                grad[i] += self.alfa * s * x ** i
            # grad[0] += learning_rate[0] * s * 1
            # grad[1] += learning_rate[1] * s * x
            # grad[2] += learning_rate[2] * s * x ** 2
            # grad[3] += learning_rate[3] * s * x ** 3
        return [component * (1.0 / n) for component in grad]

    def one_gradient_descent_step(self):
        grad = self.gradient()
        before = self.objective()
        old_bettas = self.bettas
        for index in range(self.degree):
            self.bettas[index] -= grad[index]
        after = self.objective()
        if before < after:
            self.alfa *= 0.45
            self.bettas = old_bettas
        else:
            self.alfa *= 1.05


class TwoVariableApproximator:
    def __init__(self, training_data, degree):
        self.training_data = training_data
        self.degree = degree + 1
        # the coefficients of polynomial
        self.bettas = self.get_zero_matrix()
        self.learning_rate = 0.01

    def get_zero_matrix(self):
        return [[0.0] * self.degree for _ in range(self.degree)]

    def approximation(self, double_input):
        x, y = double_input
        result = 0
        for k in range(self.degree):
            for i in range(k+1):
                b = self.bettas[k][i]
                result += b * x**(k-i) * y**i
        return result

    def objective(self):
        n = len(self.training_data)
        sum_of_squares = 0.0
        for x1, x2, y in self.training_data:
            y_hat = self.approximation([x1, x2])
            sum_of_squares += (y_hat - y) ** 2
        return sum_of_squares / (2 * n)

    def gradient(self):
        n = len(self.training_data)
        grad = self.get_zero_matrix()
        for entry in self.training_data:
            x1, x2, y = entry
            y_hat = self.approximation([x1, x2])
            for k in range(self.degree):
                for i in range(k+1):
                    grad[k][i] += (y_hat - y) * (x1 ** (k - i)) * x2 ** i

        return [[(self.learning_rate/n) * item for item in row] for row in grad]

    def one_gradient_descent_step(self):
        grad = self.gradient()
        before = self.objective()
        old_bettas = self.bettas
        for k in range(self.degree):
            for i in range(k+1):
                self.bettas[k][i] -= grad[k][i]
        after = self.objective()
        if before < after:
            self.learning_rate *= 0.45
            self.bettas = old_bettas
        else:
            self.learning_rate *= 1.05
