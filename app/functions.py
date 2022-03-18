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

    X = np.array([datapoint[0] for datapoint in datapoints])
    Y = np.array([datapoint[1] for datapoint in datapoints])

    A = np.column_stack([X ** 2, X * Y, Y ** 2, X, Y])
    b = np.ones_like(X)
    x = np.linalg.lstsq(A, b)[0].squeeze()
    return ellipse_quadratic_form_to_general_form(*x, -1)


def ellipse_quadratic_form_to_general_form(a,b,c,d,e,f):
    t = (1/2) * np.arctan(b / (a - c))
    print(t)
    cos = np.cos(t)
    sin = np.sin(t)

    a_ = a * cos**2 + b * cos * sin + c * sin**2
    b_ = 0
    c_ = a * sin**2 - b * cos * sin + c * cos**2
    d_ = d * cos + e * sin
    e_ = -d * sin + e * cos
    f_ = f

    x_0 = -d_ / (2 * a_)
    y_0 = -e_ / (2 * c_)

    a = (-f_*a_*c_ + c_*d_**2 + a_*e_**2) / (4*a_**2 * c_)
    b = (-f_*a_*c_ + c_*d_**2 + a_*e_**2) / (4*a_ * c_**2)

    return (int(x_0), int(y_0), int(np.sqrt(a)), int(np.sqrt(b)), t * 180 / np.pi)


if __name__ == "__main__":
    determine_optimal_circle([(0, 0), (1, 1), (-1, 0)])
