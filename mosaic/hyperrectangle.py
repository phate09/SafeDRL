from mosaic.interval import Interval,BoundType
from mosaic.point import Point
# from prophesy.adapter.pycarl import inf


class HyperRectangle:
    """
    Defines a hyper-rectangle, that is the Cartisean product of intervals,
    i.e. the n-dimensional variant of a box.
    """

    def __init__(self, *intervals):
        """
        :param intervals: Multiple Intervals as arguments
        """
        self.intervals = tuple(intervals)
        self._size = None

    @classmethod
    def cube(cls, left_bound, right_bound, dimension, boundtype):
        """
        
        :param left_bound: lower bound for all intervals
        :param right_bound: upper bound for all intervals
        :param dimension: dimension of the created interval
        :param boundtype: bound type for all bounds
        :return: A hyperrectangle <left_bound, right_bound>^dimension (with adequate bounds)
        """
        return cls(*[Interval(left_bound, boundtype, right_bound, boundtype) for _ in range(dimension)])

    @classmethod
    def from_extremal_points(cls, point1, point2, boundtype):
        """
        Construct a hyperrectangle from two boundary points.
        :param point1: The first point.
        :param point2: The second point.
        :param boundtype: BoundType to use as bounds for the resulting HyperRectangle.
        :return: HyperRectangle.
        """

        return cls(*[Interval(min(l, r), boundtype, max(l, r), boundtype) for l, r in zip(point1, point2)])

    def dimension(self):
        return len(self.intervals)

    def empty(self):
        for interv in self.intervals:
            if interv.empty():
                return True
        return False

    def vertices(self):
        result = []
        for i in range(0, pow(2, self.dimension()), 1):
            num_bits = self.dimension()
            bits = [(i >> bit) & 1 for bit in range(num_bits - 1, -1, -1)]
            result.append(Point(
                *[(self.intervals[i].left_bound() if x == 0 else self.intervals[i].right_bound()) for i, x in
                  zip(range(0, self.dimension()), bits)]))
        return result

    def center(self):
        return Point(*[interval.center() for interval in self.intervals])

    def get_vertex(self, pick_min_bound):
        """
        Get the vertex that corresponds to the left/right bound for the ith parameter, depending on the argument
        :param pick_min_bound: Array indicating to take the min (True) or the max (False)
        :return: 
        """
        assert len(pick_min_bound) == self.dimension()
        return Point(*[(interval.left_bound() if pmb else interval.right_bound()) for interval, pmb in zip(self.intervals, pick_min_bound)])

    def split_in_every_dimension(self):
        """
        Splits the hyperrectangle in every dimension
        
        :return: The 2^n many hyperrectangles obtained by the split
        """
        result = []
        splitted_intervals = [tuple(interv.split()) for interv in self.intervals]
        for i in range(0, pow(2, self.dimension()), 1):
            num_bits = self.dimension()
            bits = [(i >> bit) & 1 for bit in range(num_bits - 1, -1, -1)]
            result.append(HyperRectangle(*[splitted_intervals[i][x] for i, x in zip(range(0, self.dimension()), bits)]))
        return result

    def split_in_single_dimension(self, dimension):
        intervals_1 = [(interval.split()[0] if ind == dimension else interval) for ind, interval in enumerate(self.intervals)]
        intervals_2 = [(interval.split()[1] if ind == dimension else interval) for ind, interval in enumerate(self.intervals)]
        return [HyperRectangle(*intervals_1), HyperRectangle(*intervals_2)]

    def size(self):
        """
        :return: The size of the hyperrectangle
        """
        if self._size:
            return self._size
        s = 1
        for interv in self.intervals:
            s = s * interv.width()
        self._size = s
        return s

    def contains(self, point):
        """
        :param point: A Point
        :return: True if inside, False otherwise
        """
        for p, interv in zip(point, self.intervals):
            if not interv.contains(p): return False
        return True

    def intersect(self, other):
        """
        Computes the intersection
        :return:
        """
        return HyperRectangle(*[i1.intersect(i2) for i1, i2 in zip(self.intervals, other.intervals)])

    def non_empty_intersection(self, other):
        """
        
        :return: 
        """
        # TODO can be made more efficient.
        return not self.intersect(other).empty()

    def is_closed(self):
        """
        Checks whether the hyperrectangle is closed in every dimension.
        :return: True iff all intervals are closed.
        """
        for i in self.intervals:
            if not i.is_closed():
                return False
        return True

    def close(self):
        return HyperRectangle(*[i.close() for i in self.intervals])

    def _setminus(self, other, dimension):
        """
        Helper function for setminus
        :param other: 
        :param dimension: 
        :return: 
        """
        assert len(self.intervals) > dimension and len(other.intervals) > dimension
        new_interval_list = self.intervals[dimension].setminus(other.intervals[dimension])
        hrect_list = []
        if len(new_interval_list) > 1:

            # left part
            changed_interval_list = list(self.intervals)
            changed_interval_list[dimension] = new_interval_list[0]
            hrect_list.append(HyperRectangle(*changed_interval_list))

            # right part
            changed_interval_list = list(self.intervals)
            changed_interval_list[dimension] = new_interval_list[1]
            hrect_list.append(HyperRectangle(*changed_interval_list))

            # middle part which is cut away
            middle_interval = Interval(new_interval_list[0].right_bound(), new_interval_list[0].right_bound_type(),
                                       new_interval_list[1].left_bound(), new_interval_list[1].left_bound_type())
            changed_interval_list = list(self.intervals)
            changed_interval_list[dimension] = middle_interval
            hrect_list.append(HyperRectangle(*changed_interval_list))

        else:
            # the cutted box
            changed_interval_list = list(self.intervals)
            changed_interval_list[dimension] = new_interval_list[0]
            hrect_list.append(HyperRectangle(*changed_interval_list))

            # the rest which have to be cutted recursively
            changed_interval_list = list(self.intervals)
            changed_interval_list[dimension] = other.intervals[dimension]
            hrect_list.append(HyperRectangle(*changed_interval_list))

        return hrect_list

    def setminus(self, other, dimension=0):
        """
        Does a setminus operation on hyperrectangles and returns a list with hyperrects covering the resulting area
        :param other: the other HyperRectangle
        :param dimension: dimension where to start the operation
        :return: a list of HyperRectangles
        """
        assert len(other.intervals) == len(self.intervals)
        hrect_list = []
        if dimension >= len(self.intervals):
            return []
        current_rect_list = self._setminus(other, dimension)
        if len(current_rect_list) > 2:
            hrect_list.append(current_rect_list[0])
            hrect_list.append(current_rect_list[1])
            hrect_list.extend(current_rect_list[2].setminus(other, dimension + 1))
        else:
            hrect_list.append(current_rect_list[0])
            hrect_list.extend(current_rect_list[1].setminus(other, dimension + 1))
        return hrect_list

    def __str__(self):
        return " x ".join([str(i) for i in self.intervals])

    def __repr__(self):
        return "HyperRectangle({})".format(", ".join(map(repr, self.intervals)))

    def __eq__(self, other):
        for i, j in zip(self.intervals, other.intervals):
            if not i == j: return False
        return True

    def __hash__(self):
        return hash(self.intervals)

    def __len__(self):
        return len(self.intervals)

    def __iter__(self):
        return iter(self.intervals)

    def __getitem__(self, key):
        return self.intervals[key]

    def to_region_string(self, variables):
        """
        Constructs a region string, as e.g. used by storm-pars
        :param variables: An ordered list of the variables
        :type variables: List[pycarl.Variable]
        :return: 
        """
        var_strings = []
        for variable, interval in zip(variables, self.intervals):
            leftboundstr = "<=" if interval.left_bound_type() == BoundType.closed else "<"
            rightboundstr = "<=" if interval.right_bound_type() == BoundType.closed else "<"
            var_strings.append("{}{}{}{}{}".format(interval.left_bound(), leftboundstr, variable.name, rightboundstr, interval.right_bound()))
        return ",".join(var_strings)

    # @classmethod
    # def from_region_string(cls, input_string, variables):
    #     """Constructs a hyperrectangle with dimensions according to the variable order.
    #
    #     :return: A HyperRectangle
    #     """
    #     interval_strings = input_string.split(",")
    #     variables_to_intervals = dict()
    #     for int_str in interval_strings:
    #         components = int_str.split("<")
    #         if len(components) != 3:
    #             raise ValueError("Expected string in the form Number{<=,<}Variable{<=,<}Number, got {}".format(int_str))
    #
    #         if components[1][0] == "=":
    #             left_bt = BoundType.closed
    #             components[1] = components[1][1:]
    #         else:
    #             left_bt = BoundType.open
    #
    #         if components[2][0] == "=":
    #             right_bt = BoundType.closed
    #             components[2] = components[2][1:]
    #         else:
    #             right_bt = BoundType.open
    #
    #         variables_to_intervals[components[1]] = Interval(pc.Rational(components[0]), left_bt,
    #                                                          pc.Rational(components[2]), right_bt)
    #     ordered_intervals = []
    #     for variable in variables:
    #         if variable.name not in variables_to_intervals:
    #             raise RuntimeError("Parameter {} not found in region string".format(variable.name))
    #         ordered_intervals.append(variables_to_intervals[variable.name])
    #     # TODO checks.
    #     return cls(*ordered_intervals)
