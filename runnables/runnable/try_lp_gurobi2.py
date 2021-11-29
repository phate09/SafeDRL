import gurobipy as gp
import numpy as np
from gurobipy import GRB

# %%
"""Given a cloud of points, find the support function across a direction"""
# Create a new model
m = gp.Model("matrix1")
# Create variables
points = np.array([[2, 2], [2, 1.5], [1.5, 1.5], [1, 0.5]])
d = np.array([-1.0, -1.0])  # the direction

# support_value = points @ d

# minimisation trick, use a big variable to minimise (count as max but uses a min operator)
# z = m.addVar(name="z",lb=None)
z1 = m.addMVar(shape=(1,), name="z1", lb=float("-inf"))
m.setObjective(sum(z1), GRB.MINIMIZE)

# Add constraints
for i in range(len(points)):
    # m.addConstr(z >= (points @ d)[i])  # points belonging to the cloud
    m.addConstr(z1 >= (points[i] @ d))  # points belonging to the cloud

# Optimize model
m.optimize()

print(z1.X)
print('Obj: %g' % m.objVal)
