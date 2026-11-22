"""The three fourth-down Choices and the model inputs that fitting and Grading share."""

import numpy as np
from sklearn.preprocessing import SplineTransformer

FIRST_SEASON = 2015
FG_OFFSET = 18  # kick distance = yard line + 18 in nflverse (10 yards of end zone, 8 of snap)
TD_POINTS = 7
MAX_DISTANCE = 15
N_QUANTILES = 10
CHOICES = ["go", "fg", "punt"]
FIELD_ZONES = [5, 10, 20]  # conversion rates differ inside each of these
FG_KNOTS = np.array([[18.0], [30.0], [40.0], [50.0], [58.0], [66.0]])
FG_SPLINE = SplineTransformer(degree=3, knots=FG_KNOTS, extrapolation="linear").fit(FG_KNOTS)


def choice(play_type):
    """What the coach did, in the model's terms."""
    return play_type.map({"pass": "go", "run": "go", "field_goal": "fg", "punt": "punt"})


def conversion_x(ydstogo, yardline_100):
    """One rate per distance from 1 to 9 yards (10 or more share one), shifted by field zone."""
    d = np.minimum(ydstogo, 10)
    zones = [np.asarray(yardline_100) <= z for z in FIELD_ZONES]
    return np.column_stack([d == k for k in range(1, 10)] + zones)


def fg_x(distance, season):
    """A curve in distance, plus season and distance x season: kickers keep improving."""
    distance, years = np.asarray(distance, dtype=float), np.asarray(season) - 2020
    return np.column_stack([FG_SPLINE.transform(distance.reshape(-1, 1)), years, (distance - 40) * years])
