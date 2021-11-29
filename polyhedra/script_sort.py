import numpy as np

a = np.array([[1, 2, 3], [2, 1, 3], [3, 2, 1]])
print(a)
# sorted_a = a[a[:,2].argsort()] # First sort doesn't need to be stable.
# sorted_a = sorted_a[sorted_a[:,0].argsort(kind='mergesort')]
# sorted_a = sorted_a[sorted_a[:,1].argsort(kind='mergesort')]
# sorted_a=np.sort(a,axis=0)
# print(sorted_a)
sorted_indices = np.lexsort((a[:, 0], a[:, 1]), axis=0)
print(sorted_indices)
print(a[sorted_indices])
print("hello")
print()
