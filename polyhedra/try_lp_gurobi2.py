import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB

# %%
"""Given a cloud of points, find the support function across a direction"""
# Create a new model
m = gp.Model("matrix1")
# Create variables
points = np.array([[1, 1], [2, 1], [1.5, 1.5]])
d = np.array([1.0, 0])  # the direction

support_value = points @ d

# minimisation trick, use a big variable to minimise (count as max but uses a min operator)
z = m.addVar(name="z")
m.setObjective(z, GRB.MINIMIZE)

# Add constraints
for i in range(len(points)):
    m.addConstr(z >= (points @ d)[i])  # points belonging to the cloud

# Optimize model
m.optimize()

print(z.X)
print('Obj: %g' % m.objVal)
