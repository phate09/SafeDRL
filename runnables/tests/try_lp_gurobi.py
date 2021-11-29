import gurobipy as gp
import numpy as np
from gurobipy import GRB

# %%
"""Given a polyhedron (triangle) defined as halfspaces, find the furthest point across a direction"""
# Create a new model
m = gp.Model("matrix1")

# Create variables
x = m.addMVar(shape=2, name="x")
A = np.array([[0, -1], [1, 1], [-1, 0]])
b = np.array([0, 2, -1])
# Set objective
d = np.array([1.0, -1])  # the direction
m.setObjective(-d @ x, GRB.MINIMIZE)

# Add constraints
m.addConstr(A @ x <= b, name="c")  # point inside polyhedra

# Optimize model
m.optimize()

print(x.X)
print('Obj: %g' % -m.objVal)
