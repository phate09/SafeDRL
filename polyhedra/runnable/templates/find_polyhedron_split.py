from polyhedra.experiments_nn_analysis import Experiment
import numpy as np










def sample_points(template, boundaries, n_samples: int):
    pass


if __name__ == '__main__':
    template = Experiment.octagon(2)
    boundaries = (1, -1, 1, 1, 1, 1, 1, 1)

    # pick dimension
    dimension = pick_longest_dimension(template, boundaries)
    split_polyhedron(template, boundaries, dimension)
