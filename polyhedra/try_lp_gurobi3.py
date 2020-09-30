import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB

# %%
"""Given a cloud of points and a polyhedra (Ax<b), find the minimum distance across a direction"""
# Create a new model
m = gp.Model("matrix1")
# Create variables
x = m.addMVar(shape=2, name="x", lb=None)
y = m.addMVar(shape=3, name="y")
epsilon = np.array([-0.1])
# bad state polyhedra definition
A = np.array([[0, -1], [1, 1], [-1, 0]])
b = np.array([0, 2, -1])
# cloud of points
points = np.array([[2, 2], [2, 1.5], [1.5, 1.5]])
# d = m.addMVar(shape=2, name="d", lb=None)
d = np.array([-1.0, -1.0])  # the direction
z1 = m.addMVar(shape=(1,), name="z1", lb=None)

support_value = points @ d

# m.setObjective(z1+(b@y), GRB.MINIMIZE)
# Add constraints
m.addConstr(A.T @ y == -d)  # point inside polyhedra
# m.addConstr((z1 + (b@y)) <= epsilon)  # point inside polyhedra
for i in range(len(points)):
    m.addConstr(z1 >= (points[i] @ d))  # points belonging to the cloud

# Optimize model
m.optimize()

print(f"z:{z1.X}")
# print(f"d:{d.X}")
print(f"y:{y.X}")
print(f"x:{x.X}")
# print('Obj: %g' % m.objVal)
