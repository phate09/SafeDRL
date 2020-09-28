import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB

# %%
"""Given a polyhedra (triangle) defined as halfspaces, find the furthest point across a direction"""
# Create a new model
m = gp.Model("matrix1")

# Create variables
x = m.addMVar(shape=2, name="x")
A = np.array([[0, -1], [1, 1], [-1, 0]])
b = np.array([0, 2, -1])
# Set objective
obj = np.array([1.0, 0])  # the direction
m.setObjective(obj @ x, GRB.MINIMIZE)

# Add constraints
m.addConstr(A @ x <= b, name="c")  # point inside polyhedra

# Optimize model
m.optimize()

print(x.X)
print('Obj: %g' % m.objVal)
