import matplotlib.pyplot as plt


class Drawer:
    def __init__(self):
        plt.ylim((-3.0, 3.0))
        plt.xlim((0, 6.3))

    def add_points(self, points: list, c='r'):
        prev_x, prev_y = points[0]
        for x, y in points:
            drawable_point = plt.Line2D((x, x), (y, y), marker='|', color=c)
            prev_x = x
            prev_y = y
            plt.gcf().gca().add_artist(drawable_point)

    def draw(self):
        plt.show()
