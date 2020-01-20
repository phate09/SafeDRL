import os
import unittest

import jsonpickle
import scipy.spatial
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_something(self):
        K = 11
        ndata = 300000
        ndim = 4
        data = 10 * np.random.rand(ndata * ndim).reshape((ndata, ndim))
        data = data.astype(int)
        tree = scipy.spatial.cKDTree(data=data)
        result, index = tree.query([0, 0, 0, 0], k=1, p=2, n_jobs=-1)
        self.assertEqual(True, False)

    def test_kdsafe(self):
        os.chdir(os.path.expanduser("~/Development") + "/SafeDRL")
        with open("./save/t_states.json", 'r') as f:
            t_states = jsonpickle.decode(f.read())
        safe_states = t_states[0][0]
        unsafe_states = t_states[0][1]
        tree = scipy.spatial.cKDTree(data=safe_states[:, :, 0])




if __name__ == '__main__':
    unittest.main()
