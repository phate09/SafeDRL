from math import sqrt
import polyhedra.utils as utils
import matplotlib.pyplot as plt
from sympy import Point, Line, Float, Eq
from sympy.abc import x,y
import numpy as np
p1 = Point(1,1)
p2 = Point(2,1)
p3 = Point(1,0)

p4 = Point(x,y)
#%%
# plt.figure(figsize=(7, 7))
# plt.xlim(-1, 5)
# plt.ylim(-1, 5)
# plt.plot([1],[1],'o')
# plt.plot([2],[1],'o')
# # plt.plot([3*sqrt(2)/2],[3*sqrt(2)/2],'o')
# plt.plot([3/2],[3/2],'o')
# utils.newline((3/2,3/2),((2,1)))
# plt.show()
#%%
# Point(1,0).dot(Point(2,1))/Point(1,0).distance(Point(0,0))

Point.project(p2,Point(1,0))
a = np.array([[1,1],[2,1],[1.5,1.5]])
a1 = np.array([[1,1]])
a2 = np.array([[2,1]])
b = np.array([[1,0]]) #direction
b1 = np.array([[2,0]])
b2 = np.array([[1,1]])
c = utils.project(a,b2)
plt.figure(figsize=(7, 7))
plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.plot(a[:,0],a[:,1],'o')
plt.plot(c[:,0],c[:,1],'o')
# plt.plot(b2[:,0],b2[:,1],'o')
plt.show()