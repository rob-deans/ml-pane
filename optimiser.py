import numpy as np
from error import *


class GradientDescent():
    def __init__(self, cost_function):
        self.cost_function = cost_function
