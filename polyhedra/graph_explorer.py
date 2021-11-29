import numpy as np


class GraphExplorer:
    def __init__(self, template):
        # self.graph = nx.DiGraph()
        # self.dimension = int(len(template) / 2)
        # self.p = index.Property(dimension=self.dimension)
        self.root = None
        # self.coverage_tree = index.Index(interleaved=False, properties=self.p, overwrite=True)
        self.last_index = 0
        self.fringe = []
        self.seen = []

    # def store_boundary(self, boundary: tuple):
    #     boundary = tuple(np.array(boundary).round(4))  # rounding to 4 decimal digits
    #     covered = self.is_covered_simple(boundary) is not None  # self.is_covered(boundary)
    #     if not covered:
    #         self.graph.add_node(boundary)
    #         self.last_index += 1  # self.coverage_tree.insert(self.last_index, self.convert_boundary_to_rtree_boundary(boundary[0:12]), boundary)
    #         return True
    #     return False

    def store_in_fringe(self, boundary: tuple):
        boundary = tuple(np.array(boundary).round(4))  # rounding to 4 decimal digits
        covered = self.is_covered_seen(boundary)
        if not covered:
            covered = self.is_covered_fringe(boundary)
            if not covered:
                if len(self.fringe) != 0:
                    # remove old polyhedra contained in the new polyhedron
                    self.fringe = [old for old in self.fringe if not self.compare_polyhedra(boundary, old)]
                # append new polyhedron at the end
                self.fringe.append(boundary)
                return True
        return False

    def archive_fringe(self):
        for boundary in self.fringe:
            self.seen.append(boundary)
        old_fringe = self.fringe
        self.fringe = []
        return old_fringe

    @staticmethod
    def convert_boundary_to_rtree_boundary(boundary: tuple):
        boundary_array = np.array(boundary)
        boundary_array = np.abs(boundary_array)
        return tuple(boundary_array)

    def is_covered_seen(self, other: tuple):
        for x in self.seen:
            contained = self.compare_polyhedra(x, other)
            if contained:
                return True
        return False

    def is_covered_fringe(self, other: tuple):
        for x in self.fringe:
            contained = self.compare_polyhedra(x, other)
            if contained:
                return True
        return False

    @staticmethod
    def compare_polyhedra(x: tuple, other: tuple):
        assert len(x) == len(other)
        contained = True
        for i in range(len(x)):
            if x[i] < other[i]:
                contained = False
                break
        return contained

    # def get_next_in_fringe(self):
    #     min_distance = float("inf")
    #     result = defaultdict(list)
    #     shortest_path = nx.shortest_path(self.graph, source=self.root)
    #     for key in shortest_path:
    #         attr: dict = self.graph.nodes[key]
    #         if self.graph.out_degree[key] == 0 and (not attr.keys().__contains__("ignore") or attr["ignore"] == False):
    #             distance = len(shortest_path[key])
    #             if distance < min_distance:
    #                 min_distance = distance
    #             result[distance].append(key)
    #     return result[min_distance]
