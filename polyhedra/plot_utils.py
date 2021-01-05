from typing import List
import plotly.graph_objects as go
import numpy as np
from mosaic.utils import compute_trace_polygons, PolygonSort


def transform_vertices(polygon_vertices_list):
    result = []
    for vertices in polygon_vertices_list:
        transformed_vertices = []
        for vertex in vertices:
            transformed_vertex = np.zeros(shape=(2,))
            transformed_vertex[0] = vertex[0] - vertex[1]
            transformed_vertex[1] = vertex[3]
            transformed_vertices.append(transformed_vertex)
        result.append(transformed_vertices)
    return result


def transform_vertices2(polygon_vertices_list):
    result = []
    for vertices in polygon_vertices_list:
        transformed_vertices = []
        for vertex in vertices:
            transformed_vertex = np.zeros(shape=(2,))
            transformed_vertex[0] = vertex[0]
            transformed_vertex[1] = vertex[1]
            transformed_vertices.append(transformed_vertex)
        result.append(transformed_vertices)
    return result


def show_polygon_list(polygon_vertices_list):  # x_prime_vertices, x_vertices

    # scaler = StandardScaler()
    # scaler.fit(polygon_vertices_list[0])  # list(itertools.chain.from_iterable(polygon_vertices_list)))
    # scaled_list = []
    # for x in polygon_vertices_list:
    #     scaled_list.append(scaler.transform(x))
    # pca = PCA(n_components=2)
    # pca.fit(list(itertools.chain.from_iterable(scaled_list)))  #
    # principal_components_list = []
    # for x_scaled in scaled_list:
    #     principal_components_list.append(pca.transform(x_scaled))
    traces = []
    for timestep in polygon_vertices_list:
        principal_components_list = transform_vertices(timestep)
        traces.append(compute_polygon_trace(principal_components_list))
    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)
    fig.update_layout(xaxis_title="x_lead - x_ego", yaxis_title="Speed")
    fig.show()


def show_polygon_list2(polygon_vertices_list, y_axis_title="x_ego", x_axis_title="x_lead"):
    traces = []
    for timestep in polygon_vertices_list:
        principal_components_list = transform_vertices2(polygon_vertices_list[timestep])
        traces.append(compute_polygon_trace(principal_components_list))
    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)
    fig.update_layout(xaxis_title=x_axis_title, yaxis_title=y_axis_title)
    fig.show()


def compute_polygon_trace(principalComponents: List[List]):
    polygon1 = [PolygonSort(x) for x in principalComponents]
    trace1 = compute_trace_polygons(polygon1)
    return trace1
