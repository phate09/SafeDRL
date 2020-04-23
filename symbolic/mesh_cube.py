from typing import Tuple, List

import plotly.graph_objects as go
import numpy as np
from sympy.combinatorics import GrayCode
import plotly.express as px


def get_mesh(domain: Tuple[Tuple[float, float]], color: str):
    x, y, z = get_coordinates(domain)
    return go.Mesh3d(x=x, y=y, z=z, alphahull=1, opacity=0.1, color=color)


def get_coordinates(domain: Tuple[Tuple[float, float]]):
    state_size = len(domain)
    a = GrayCode(state_size)
    codes = list(a.generate_gray())
    # codes.append(codes[0])
    x = []
    y = []
    z = []
    for code in codes:
        for d in range(state_size):
            value = int(code[d])
            if domain[d][0] == domain[d][1] and value == 1:
                domain_measure = domain[d][value] + 0.001
            else:
                domain_measure = domain[d][value]
            if d == 0:
                z.append(domain_measure)
            elif d == 1:
                x.append(domain_measure)
            else:
                y.append(domain_measure)
    return x, y, z
