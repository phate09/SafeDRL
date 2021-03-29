import numpy as np
import pypoman
points = np.random.rand(30, 2)  # 30 random points in 2-D

vertices = pypoman.duality.convex_hull(points)
A,b = pypoman.duality.compute_polytope_halfspaces(vertices)