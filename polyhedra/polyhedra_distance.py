import numpy as np
import scipy.sparse as sp
import gurobipy as gp


def is_separable(points: np.ndarray, A, b):
    m = gp.Model("matrix1")
    # Create variables
    y = m.addMVar(shape=(3,), name="y", lb=0)
    epsilon = np.array([-1])
    # bad state polyhedra definition

    # cloud of points
    # points = np.array([[2, 2], [2, 1.5], [1.5, 1.5]])  # , [1, 0.5]
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
        return True
    elif m.Status == gp.GRB.INFEASIBLE:
        print("Model infeasible")
        return False
    else:
        print(f"Unknown code: {m.Status}")
        return False


if __name__ == '__main__':
    points = np.array([[2, 2], [2, 1.5], [1.5, 1.5], [1, 0.5]])  #
    A = np.array([[0, -1], [1, 1], [-1, 0]])
    b = np.array([0, 2, -1])
    is_separable(points, A, b)
