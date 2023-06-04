"""
Module containing utility functions for the project.
"""

import numpy as np


def euclidean_distance(x_1: float, y_1: float, x_2: float, y_2: float) -> float:
    """
    Returns euclidean distance between two points.

    Args:
        x1 (float): x coordinate of first point
        y1 (float): y coordinate of first point
        x2 (float): x coordinate of second point
        y2 (float): y coordinate of second point

    Returns:
        float: euclidean distance between two points
    """
    return np.linalg.norm(np.array([x_1, y_1]) - np.array([x_2, y_2]))
