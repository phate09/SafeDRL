import decimal
import math
import operator
import shelve
from collections import defaultdict
from functools import reduce
from typing import Tuple, List

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import intervals as I
import numpy as np
import ray
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from plotly import express as px
from rtree import index
import plotly.express as px
import importlib

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from symbolic.mesh_cube import get_mesh
from networkx.drawing.nx_pydot import write_dot
from colour import Color


def array_to_tuple(array: np.ndarray) -> Tuple[Tuple[float, float]]:
    array_of_tuples = map(tuple, array)
    tuple_of_tuples = tuple(array_of_tuples)
    return tuple_of_tuples


def area_tensor(domain: torch.Tensor) -> float:
    '''
    Compute the area of the domain
    '''
    dom_sides = domain.select(1, 1) - domain.select(1, 0)
    dom_area = dom_sides.prod()
    return abs(float(dom_area.item()))


def area_numpy(domain: np.ndarray) -> float:
    '''
    Compute the area of the domain
    '''
    dom = np.array(domain)
    dom_sides = dom[:, 1] - dom[:, 0]
    dom_area = dom_sides.prod()
    return abs(float(dom_area.item()))


def area_tuple(domain: Tuple[Tuple[float, float]]):
    dimensions = [abs(x[1] - x[0]) for x in domain]
    area = reduce(operator.mul, dimensions, 1)
    return area


def centre_tuple(domain: Tuple[Tuple[float, float]]) -> Tuple[float]:
    centre = tuple([(x[1] + x[0]) / 2 for x in domain])
    return centre


@ray.remote
def filter_helper(interval_to_fill, current_interval):
    """Check if the interval_to_fill is overlaps current_interval and returns a trimmed version of it"""
    contains = interval_contains(interval_to_fill, current_interval)
    return shrink(interval_to_fill, current_interval) if contains else None


def shrink(a, b):
    """Shrink interval a to be at max as big as b"""
    dimensions = len(a)
    state = tuple([(max(a[dimension][0], b[dimension][0]), min(a[dimension][1], b[dimension][1])) for dimension in range(dimensions)])
    return state


def interval_contains(a, b):
    """Condition used to check if an a touches b and partially covers it"""
    dimensions = len(a)
    partial = all(
        [(I.closed(*a[dimension]) & I.open(*b[dimension])).is_empty() == False if not I.open(*b[dimension]).is_empty() else (I.closed(*a[dimension]) & I.closed(*b[dimension])).is_empty() == False for
         dimension in range(dimensions)])
    return partial


def contained(a: tuple, b: tuple):
    return b[0] <= a[0] <= b[1] and b[0] <= a[1] <= b[1]


def partially_contained(a: tuple, b: tuple):
    return b[0] <= a[0] <= b[1] or b[0] <= a[1] <= b[1]


def partially_contained_interval(a: tuple, b: tuple):
    return all([b[dimension][0] <= a[dimension][0] <= b[dimension][1] or b[dimension][0] <= a[dimension][1] <= b[dimension][1] for dimension in range(len(a))])


def non_zero_area(a: tuple):
    return all([abs(bounds[0] - bounds[1]) != 0 for bounds in a])


def beep():
    import os
    duration = 0.5  # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))


def shelve_variables2():
    my_shelf = shelve.open('/tmp/shelve.out', 'n')  # 'n' for new

    for key in globals():
        try:
            my_shelf[key] = globals()[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))
        except:
            print('GENERIC ERROR shelving: {0}'.format(key))
    my_shelf.close()


def unshelve_variables():
    my_shelf = shelve.open('/tmp/shelve.out')
    for key in my_shelf:
        globals()[key] = my_shelf[key]
    my_shelf.close()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def chunker_list(seq, size):
    "splits the list is size sublists"
    return (seq[i::size] for i in range(size))


def round_tuples(intervals: List[Tuple[Tuple[Tuple[float, float]], bool]], rounding: int) -> List[Tuple[Tuple[Tuple[float, float]], bool]]:
    return [(round_tuple(interval, rounding), action) for interval, action in intervals]


def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


def round_tuple(interval: Tuple[Tuple[float, float]], rounding: int) -> Tuple[Tuple[float, float]]:
    return tuple([(float(round(x[0], rounding)), float(round(x[1], rounding))) for x in interval])


# def inflate(current_interval: Tuple[Tuple[float, float]], rounding: int, eps=1e-6, ) -> Tuple[Tuple[float, float]]:
#     current_interval = round_tuple(tuple([(x[0] - eps, x[1] + eps) for x in current_interval]), rounding)  # rounding
#     return current_interval


def flatten_interval(current_interval: Tuple[Tuple[float, float]]) -> Tuple:
    result = []
    for d in range(len(current_interval)):
        result.extend([current_interval[d][0], current_interval[d][1]])
    return tuple(result)


def show_plot3d(*args):
    meshes = []
    for i, interval_list in enumerate(args):
        x_list = []
        y_list = []
        if len(interval_list) == 0:
            continue
        if count_elements(interval_list[0]) % 2 != 0:
            interval_list = [x[0] for x in interval_list]  # remove the action component from the list
        color = str(px.colors.qualitative.Plotly[i])
        for interval in interval_list:
            # fig.add_mesh3d(get_mesh(interval, color))
            meshes.append(get_mesh(interval, color))
    fig = go.Figure(data=meshes)
    fig.show()


def show_plot(*args, legend: List = None):
    fig = go.Figure()
    for i, interval_list in enumerate(args):
        x_list = []
        y_list = []
        if len(interval_list) == 0:
            continue
        if count_elements(interval_list[0]) % 2 != 0:
            interval_list = [x[0] for x in interval_list]  # remove the action component from the list
        for interval in interval_list:
            x = [interval[0][0], interval[0][1], interval[0][1], interval[0][0], interval[0][0]]
            y = [interval[1][0], interval[1][0], interval[1][1], interval[1][1], interval[1][0]]
            x_list.extend(x)
            x_list.append(None)
            y_list.extend(y)
            y_list.append(None)
        name = legend[i] if legend is not None and len(legend) > i else None
        fig.add_scatter(x=x_list, y=y_list, fill="toself", hoveron="points", name=name)
    fig.update_shapes(dict(xref='x', yref='y'))
    # fig['layout']['yaxis1'].update(title='', autorange=False)
    # fig['layout']['xaxis1'].update(title='', autorange=False)
    # fig.update_layout(autosize=False)
    fig.show()
    return fig


def p_chart(interval_list: List[Tuple[Tuple[Tuple[float, float]], float]], *, title=None, save_to: str = None, rounding=4):
    fig: go.Figure = go.Figure()
    if len(interval_list) == 0:
        return
    colors = list(Color("blue").range_to(Color("red"), (10 ** rounding) + 1))
    probabilities = set(map(lambda x: round(x[1], rounding), interval_list))  # round to 4 digits
    newlist = [(x, [y[0] for y in interval_list if round(y[1], rounding) == x]) for x in sorted(probabilities)]
    areas = []
    probabilities = []
    total_area = sum([area_tuple(y[0]) for y in interval_list])
    for probability, intervals in newlist:
        x_list = []
        y_list = []
        for interval in intervals:
            x = [interval[0][0], interval[0][1], interval[0][1], interval[0][0], interval[0][0]]
            y = [interval[1][0], interval[1][0], interval[1][1], interval[1][1], interval[1][0]]
            x_list.extend(x)
            x_list.append(None)
            y_list.extend(y)
            y_list.append(None)
        value_int = int(round(probability, rounding) * (10 ** rounding))
        color = colors[value_int].get_hex_l()
        # fig.add_scatter(x=x_list, y=y_list, fill="toself", fillcolor=color, line={"color": color}, opacity=0.2, name=f"{probability:.{rounding}f}", marker=dict(size=1))  # hoveron="points"
        area = sum([area_tuple(x) for x in intervals])
        areas.append(area)
        probabilities.append(str(probability))
        fig.add_bar(x=[probability], y=[area / total_area], marker_color=color, name=f"{probability:.{rounding}f}")
    # for interval, probability in interval_list:
    #     x = [interval[0][0], interval[0][1], interval[0][1], interval[0][0], interval[0][0]]
    #     y = [interval[1][0], interval[1][0], interval[1][1], interval[1][1], interval[1][0]]
    #     color = assign_color(probability)
    #     fig.add_scatter(x=x, y=y, fill="toself", fillcolor=color, opacity=0.2, line=dict(color=color), name=str(probability), hovertext=str(probability), marker=dict(size=0))
    margin = go.layout.Margin(l=0,  # left margin
                              r=0,  # right margin
                              b=0,  # bottom margin
                              t=0  # top margin
                              )
    fig.update_layout(margin=margin, xaxis_type="category")  # ,yaxis_type="log"
    if title is not None:
        fig.update_layout(title=title, title_x=0.5)
    fig.show()
    if save_to is not None:
        fig.write_image(save_to, width=800, height=800)


def show_heatmap(interval_list: List[Tuple[Tuple[Tuple[float, float]], float]], *, title=None, save_to: str = None, rounding=4):
    fig: go.Figure = go.Figure()
    if len(interval_list) == 0:
        return
    colors = list(Color("blue").range_to(Color("red"), (10 ** rounding) + 1))
    probabilities = set(map(lambda x: round(x[1], rounding), interval_list))  # round to 4 digits
    newlist = [(x, [y[0] for y in interval_list if round(y[1], rounding) == x]) for x in sorted(probabilities)]
    for probability, intervals in newlist:
        x_list = []
        y_list = []
        for interval in intervals:
            x = [interval[0][0], interval[0][1], interval[0][1], interval[0][0], interval[0][0]]
            y = [interval[1][0], interval[1][0], interval[1][1], interval[1][1], interval[1][0]]
            x_list.extend(x)
            x_list.append(None)
            y_list.extend(y)
            y_list.append(None)
        value_int = int(round(probability, rounding) * (10 ** rounding))
        color = colors[value_int].get_hex_l()
        fig.add_scatter(x=x_list, y=y_list, fill="toself", fillcolor=color, line={"color": color}, opacity=0.2, name=f"{probability:.{rounding}f}", marker=dict(size=1))  # hoveron="points"
    # for interval, probability in interval_list:
    #     x = [interval[0][0], interval[0][1], interval[0][1], interval[0][0], interval[0][0]]
    #     y = [interval[1][0], interval[1][0], interval[1][1], interval[1][1], interval[1][0]]
    #     color = assign_color(probability)
    #     fig.add_scatter(x=x, y=y, fill="toself", fillcolor=color, opacity=0.2, line=dict(color=color), name=str(probability), hovertext=str(probability), marker=dict(size=0))
    margin = go.layout.Margin(l=0,  # left margin
                              r=0,  # right margin
                              b=0,  # bottom margin
                              t=0  # top margin
                              )
    fig.update_layout(margin=margin)
    if title is not None:
        fig.update_layout(title=title, title_x=0.5)
    fig.show()
    if save_to is not None:
        fig.write_image(save_to, width=800, height=800)


def count_elements(l):
    count = 0
    for item in l:
        if isinstance(item, tuple):
            count += count_elements(item)
        else:
            count += 1
    return count

    # def show_plot(intervals_action: List[Tuple[Tuple[Tuple[float, float]], bool]] = None, intervals: List[Tuple[Tuple[float, float]]] = None, aggregate=True):  #     fig = go.Figure()  #     x_y_dict = defaultdict(list)  #     if intervals_action is None:  #         intervals_action = []  #     if intervals is None:  #         intervals = []  #     intervals_with_action = [(x, None) for x in intervals]  #     for interval in intervals_with_action + intervals_action:  #         if interval[1] is True:  #             color = 'Red'  #         elif interval[1] is False:  #             color = 'Blue'  #         elif interval[1] is None:  #             color = 'Green'  #         else:  #             color = interval[1]  #         x = [interval[0][0][0], interval[0][0][1], interval[0][0][1], interval[0][0][0], interval[0][0][0]]  #         y = [interval[0][1][0], interval[0][1][0], interval[0][1][1], interval[0][1][1], interval[0][1][0]]  #         x_y_dict[color].append((x, y))  #     if not aggregate:  #         for color in x_y_dict.keys():  #             for x, y in x_y_dict[color]:  #                 fig.add_scatter(x=x, y=y, fill="toself", fillcolor=color)  #     else:  #         for color in x_y_dict.keys():  #             x_list = []  #             y_list = []  #             for x, y in x_y_dict[color]:  #                 x_list.extend(x)  #                 x_list.append(None)  #                 y_list.extend(y)  #                 y_list.append(None)  #             fig.add_scatter(x=x_list, y=y_list, fill="toself", fillcolor=color)  #     fig.update_shapes(dict(xref='x', yref='y'))  #     fig.show()


def create_tree(intervals: List[Tuple[Tuple[Tuple[float, float]], bool]]) -> index.Index:
    if len(intervals) != 0:
        state_size = len(intervals[0][0])
        p = index.Property(dimension=state_size)
        helper = bulk_load_rtree_helper(intervals)
        tree = index.Index(helper, interleaved=False, properties=p, overwrite=True)
        return tree
    else:
        raise Exception("len(intervals) cannot be 0")


def bulk_load_rtree_helper(data: List[Tuple[Tuple[Tuple[float, float]], bool]]):
    for i, obj in enumerate(data):
        interval = obj[0]
        yield (i, flatten_interval(interval), obj)


def save_graph_as_dot(graph):
    pos = nx.nx_agraph.graphviz_layout(graph)
    nx.draw(graph, pos=pos)
    write_dot(graph, 'file.dot')
    replace_list = []
    replace_list.append(("fail=True", "color = orange, penwidth=4.0"))
    replace_list.append(("p=\"", "xlabel=\""))
    replace_list.append(("lb=\"", "xlabel=\""))
    replace_list.append(("strict digraph  {", "strict digraph  { ranksep=4;"))
    inplace_change('file.dot', replace_list)


def inplace_change(filename, replace_list: List[Tuple[str, str]]):
    # Safely read the input filename using 'with'
    with open(filename) as f:
        s = f.read()

    # Safely write the changed content, if found in the file
    with open(filename, 'w') as f:
        # s = f.read()
        for old, new in replace_list:
            s = s.replace(old, new)
        f.write(s)


def pca_map(results, save_path, state_size):
    data = [tuple([x for x in centre]) + (area, action, prob) for (interval, centre, area, action, prob) in results]
    main_frame: pd.DataFrame = pd.DataFrame(data)
    x = StandardScaler().fit_transform(main_frame.iloc[:, 0:state_size])
    pca = PCA(n_components=2)
    pca.fit(x)
    principalComponents = pca.transform(x)
    pca_pd = pd.DataFrame(principalComponents, columns=["A", "B"])
    pca_pd.insert(2, "area", main_frame.iloc[:, state_size])
    pca_pd.insert(3, "prob", main_frame.iloc[:, state_size + 2])
    fig = px.scatter(pca_pd, x="A", y="B", size="area", color="prob", size_max=100, height=1200)  # ,animation_frame="t"
    margin = go.layout.Margin(l=0,  # left margin
                              r=0,  # right margin
                              # b=0,  # bottom margin
                              t=0  # top margin
                              )
    fig.update_layout(margin=margin)
    fig.write_image(save_path, width=800, height=800)
    fig.show()
    return fig
