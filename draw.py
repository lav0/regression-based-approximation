import matplotlib.pyplot as plt


class Drawer:
    def __init__(self, xlimits=(0.0, 6.3), ylimits=(-3.0, 3.0)):
        self.figure_number = figure_number()
        plt.figure(self.figure_number)
        plt.ylim(ylimits)
        plt.xlim(xlimits)

    def add_points(self, points: list, c='r'):
        plt.figure(self.figure_number)
        for x, y in points:
            drawable_point = plt.Line2D((x, x), (y, y), marker='|', color=c)
            plt.gcf().gca().add_artist(drawable_point)

    def draw(self):
        plt.figure(self.figure_number)
        plt.show()


figure_number_factory = 1


def figure_number():
    global figure_number_factory
    number = figure_number_factory
    figure_number_factory += 1
    return number
