"""Provide implementation of the Search-For-Dividing-Circle by the perceptron."""
import accessify
import numpy as np

from SearchForDividingCircle.DividingCircle.interfaces import DividingCircleInterface


@accessify.implements(DividingCircleInterface)
class DividingCircle:
    """Implementation of the Search-For-Dividing-Circle by the perceptron."""

    def __init__(self, radius_range: tuple, center_circle_start_coordinates: dict,
                 center_circle_end_coordinates: dict, amount_points=6000, accuracy=1e-4):
        """Docs."""
        self.radius_range_start, self.radius_range_end = radius_range
        self.center_circle_start_coordinates = center_circle_start_coordinates
        self.center_circle_end_coordinates = center_circle_end_coordinates
        self.amount_points = amount_points
        self.accuracy = accuracy
        self.area_coordinates_start, self.area_coordinates_end = self._get_area_coordinates()
        self.area_range_x = self.area_coordinates_end.get('x') - self.area_coordinates_start.get('x')
        self.area_range_y = self.area_coordinates_end.get('y') - self.area_coordinates_start.get('y')
        self.center_coordinates = self._get_center_coordinates()
        self.radius = self._get_radius()
        self.points = None
        self.inner_points = None
        self.outer_points = None

    def generate_points(self):
        """Docs"""
        points_x = np.array([])
        points_y = np.array([])
        for i in range(self.amount_points):
            points_x = np.append(points_x, np.random.rand() * self.area_range_x + self.area_coordinates_start.get('x'))
            points_y = np.append(points_y, np.random.rand() * self.area_range_y + self.area_coordinates_start.get('y'))

        points_x = points_x.reshape(-1, 1)
        points_y = points_y.reshape(-1, 1)

        self.points = np.concatenate([points_x, points_y], axis=1)
        self._random_divide_points(self.points)

    def _random_divide_points(self, points):
        """(x-a)^2 + (y-b)^2 < R^2(+-)accuracy"""
        circle_center_x = (points[:, 0] - self.center_coordinates.get('x'))
        circle_center_y = (points[:, 1] - self.center_coordinates.get('y'))

        inner_points = points[circle_center_x ** 2 + circle_center_y ** 2 < self.radius ** 2 - self.accuracy, :]
        outer_points = points[circle_center_x ** 2 + circle_center_y ** 2 > self.radius ** 2 + self.accuracy]
        outer_points = outer_points[:inner_points.shape[0], :]

        self.inner_points, self.outer_points = inner_points, outer_points
        self.x_train = np.concatenate([inner_points, outer_points], axis=0)
        self.y_train = np.concatenate([np.zeros((inner_points.shape[0],)),np.ones((outer_points.shape[0]))])

    def _get_center_coordinates(self):
        center_x_norm = np.random.rand()
        center_y_norm = np.random.rand()
        center_x_range = self.center_circle_end_coordinates.get('x') - self.center_circle_start_coordinates.get('x')
        center_y_range = self.center_circle_end_coordinates.get('y') - self.center_circle_start_coordinates.get('y')
        return {
            'x': center_x_norm * center_x_range + self.center_circle_start_coordinates.get('x'),
            'y': center_y_norm * center_y_range + self.center_circle_start_coordinates.get('y'),
        }

    def _get_radius(self):
        radius_norm = np.random.rand()
        return radius_norm * (self.radius_range_end - self.radius_range_start) + self.radius_range_start

    def _get_area_coordinates(self):
        radius_length = self.radius_range_end - self.radius_range_start
        area_coordinates_start = {
            'x': self.center_circle_start_coordinates.get('x') - radius_length,
            'y': self.center_circle_start_coordinates.get('y') - radius_length,
        }
        area_coordinates_end = {
            'x': self.center_circle_end_coordinates.get('x') + radius_length,
            'y': self.center_circle_end_coordinates.get('y') + radius_length,
        }
        return area_coordinates_start, area_coordinates_end

    def train(self):
        """https://medium.com/@thomascountz/19-line-line-by-line-python-perceptron-b6f113b161f3"""
        perceptrone_params = np.array([0., 1., 0., 0.])
        is_end_array = False
        print("------------------Start of perceptron training---------------------")
        while not is_end_array:
            for i, (dots, label) in enumerate(zip(self.x_train, self.y_train)):
                x = np.array([1, dots[0] ** 2 + dots[1] ** 2, dots[0], dots[1]])

                if perceptrone_params.dot(x.T) > 0 and label == 0:
                    perceptrone_params -= x
                    break

                if perceptrone_params.dot(x.T) < 0 and label == 1:
                    perceptrone_params += x
                    break

                if i == self.x_train.shape[0] - 1:
                    is_end_array = True

        print("------------------End of perceptron training---------------------\n")
        return perceptrone_params
