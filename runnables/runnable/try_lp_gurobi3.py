import gurobipy as gp
import numpy as np

# %%
"""Given a cloud of points and a polyhedron (Ax<b), find the minimum distance across a direction"""
# Create a new model
m = gp.Model("matrix1")
# Create variables
y = m.addMVar(shape=(3,), name="y", lb=0)
epsilon = np.array([-1])
# bad state polyhedra definition
A = np.array([[0, -1], [1, 1], [-1, 0]])
b = np.array([0, 2, -1])
# cloud of points
points = np.array([[2, 2], [2, 1.5], [1.5, 1.5]])  # , [1, 0.5]
d = m.addMVar(shape=(2,), name="d", lb=float("-inf"))
# d = np.array([-1.0, -1.0])  # the direction vector
z1 = m.addMVar(shape=(1,), name="z1", lb=float("-inf"))

# m.setObjective(sum(z1)+(b @ y), GRB.MINIMIZE)  #- (b @ y)sum(z1)
# Add constraints
m.addConstr(A.T @ y == -d)  # fix direction as opposite of d
m.addConstr((z1 + (b @ y)) <= epsilon)  # point inside polyhedra
for i in range(len(points)):
    m.addConstr(z1 >= (points[i] @ d))  # points belonging to the cloud

# Optimize model
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print("Model solved successfully")
elif m.Status == gp.GRB.INFEASIBLE:
    print("Model infeasible")
else:
    print(f"Unknown code: {m.Status}")
print(f"z:{z1.X}")
print(f"y:{y.X}")
print(f"b@y:{(b @ y.X)}")
print(f"d:{d.X}")
print("finish")  # print(f"x:{x.X}")  # print('Obj: %g' % m.objVal)
