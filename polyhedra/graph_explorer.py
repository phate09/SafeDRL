from collections import defaultdict

import networkx as nx
import numpy as np
from rtree import index


class GraphExplorer:
    def __init__(self, template):
        self.graph = nx.DiGraph()
        self.dimension = int(len(template) / 2)
        self.p = index.Property(dimension=self.dimension)
        self.root = None
        self.coverage_tree = index.Index(interleaved=False, properties=self.p, overwrite=True)
        self.last_index = 0

    def store_boundary(self, boundary: tuple):
        boundary = tuple(np.array(boundary).round(4))  # rounding to 4 decimal digits
        covered = self.is_covered(boundary)
        if not covered:
            self.graph.add_node(boundary)
            self.last_index += 1
            self.coverage_tree.insert(self.last_index, self.convert_boundary_to_rtree_boundary(boundary), boundary)

    @staticmethod
    def convert_boundary_to_rtree_boundary(boundary: tuple):
        boundary_array = np.array(boundary)
        boundary_array = np.abs(boundary_array)
        return tuple(boundary_array)

    def is_covered(self, other: tuple):
        intersection = self.coverage_tree.intersection(self.convert_boundary_to_rtree_boundary(other), objects='raw')
        for item in intersection:
            contained = True
            for i, element in enumerate(item):
                other_element = other[i]
                if i % 2 == 0:
                    if element < other_element:
                        contained = False
                        break
                else:
                    if element > other_element:
                        contained = False
                        break
            if contained:
                return True
        return False

    def get_next_in_fringe(self):
        min_distance = float("inf")
        result = defaultdict(list)
        shortest_path = nx.shortest_path(self.graph, source=self.root)
        for key in shortest_path:
            if self.graph.out_degree[key] == 0:
                distance = len(shortest_path[key])
                if distance < min_distance:
                    min_distance = distance
                result[distance].append(key)
        return result[min_distance]
