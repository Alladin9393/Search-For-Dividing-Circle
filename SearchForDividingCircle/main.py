"""Provide Search-For-Dividing-Circle by the perceptron."""

import numpy as np

from SearchForDividingCircle.DividingCircle import DividingCircle

if __name__ == '__main__':
    center_circle_start_coordinates = {
        'x': -10,
        'y': -10,
    }
    center_circle_end_coordinates = {
        'x': 10,
        'y': 10,
    }

    DC = DividingCircle(
        radius_range=(1, 7),
        center_circle_start_coordinates=center_circle_start_coordinates,
        center_circle_end_coordinates=center_circle_end_coordinates,
        amount_points=1500,
    )

    DC.generate_points()
    print(DC.points.shape)
    print(DC.inner_points.shape)
    print(DC.outer_points.shape)
    print(DC.x_train.shape)
    print(DC.y_train.shape)
    perceptron_params = DC.train()

    print("center_x_coordinate : {} , center_y_coordinates : {}, radius : {}".format(
        DC.center_coordinates.get('x'),
        DC.center_coordinates.get('y'),
        DC.radius,
    ))

    center_x_coordinate_prediction = (-1 / 2) * (perceptron_params[2] / perceptron_params[1])
    center_y_coordinate_prediction = (-1 / 2) * (perceptron_params[3] / perceptron_params[1])
    radius_prediction = np.sqrt(center_x_coordinate_prediction ** 2 + center_y_coordinate_prediction ** 2
                                - perceptron_params[0] / perceptron_params[1])

    print("\ncenter_x_coordinate_prediction : {} , center_y_coordinate_prediction : {}, radius_prediction : {}".format(
        center_x_coordinate_prediction,
        center_y_coordinate_prediction,
        radius_prediction,
    ))
