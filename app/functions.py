import typing
import numpy as np


def determine_optimal_circle(
    datapoints: typing.List[typing.Tuple[int]],
) -> typing.Tuple[float]:
    """_summary_

    Returns the following coefficients:
    a*x**2 + c*x*y + b*y**2 + d*x + e*y -1 = 0

    Args:
        datapoints (typing.List[typing.Tuple[int]]): _description

    Returns:
        ellipse (typing.Tuple[float]):
    """

    number_of_datapoints = len(datapoints)
    if number_of_datapoints < 20:
        return 1, 1, 1, 1, 1

    X = np.array([datapoint[0] for datapoint in datapoints]).reshape(
        (number_of_datapoints, 1)
    )
    Y = np.array([datapoint[1] for datapoint in datapoints]).reshape(
        (number_of_datapoints, 1)
    )

    A = np.hstack([X ** 2, X * Y, Y ** 2, X, Y])
    b = np.ones_like(X)
    x = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()

    return ellipse_quadratic_form_to_general_form([*x, -1])


def ellipse_quadratic_form_to_general_form(coeffs):
    print(coeffs)
    return


if __name__ == "__main__":
    determine_optimal_circle([(0, 0), (1, 1), (-1, 0)])
