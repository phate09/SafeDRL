import numpy as np


class Scheduler:
    def __init__(self, start, end, steps):
        self.end = end
        self.start = start
        self.steps = steps
        self.progression = np.linspace(start, end, steps)

    def get(self, item):
        if item < self.steps:
            return self.progression[item]
        else:
            return self.end
