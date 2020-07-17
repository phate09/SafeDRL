from typing import Tuple, List
from mosaic.interval import Interval, BoundType
from mosaic.point import Point


# from prophesy.adapter.pycarl import inf


class HyperRectangle:
    """
    Defines a hyper-rectangle, that is the Cartisean product of intervals,
    i.e. the n-dimensional variant of a box.
    """

    def __init__(self, *intervals: Interval):
        """
        :param intervals: Multiple Intervals as arguments
        """
        self.intervals: Tuple[Interval] = tuple(intervals)
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
        return cls(*[Interval(left_bound, right_bound, boundtype, boundtype) for _ in range(dimension)])

    @classmethod
    def from_extremal_points(cls, point1, point2, boundtype):
        """
        Construct a hyperrectangle from two boundary points.
        :param point1: The first point.
        :param point2: The second point.
        :param boundtype: BoundType to use as bounds for the resulting HyperRectangle.
        :return: HyperRectangle.
        """

        return cls(*[Interval(min(l, r), max(l, r), boundtype, boundtype) for l, r in zip(point1, point2)])

    @classmethod
    def from_numpys(cls, array):
        new_items = []
        for item in array:
            new_items.append(cls.from_numpy(item))
        return new_items

    @classmethod
    def from_numpy(cls, array):
        return cls(*[Interval(array[0][i], array[1][i]) for i in range(array.shape[-1])])

    @classmethod
    def from_tuple(cls, interval):
        return cls(*[Interval(interval[i][0], interval[i][1]) for i in range(len(interval))])

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
            result.append(Point(*[(self.intervals[i].left_bound() if x == 0 else self.intervals[i].right_bound()) for i, x in zip(range(0, self.dimension()), bits)]))
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

    def split_in_every_dimension(self, rounding: int):
        """
        Splits the hyperrectangle in every dimension
        
        :return: The 2^n many hyperrectangles obtained by the split
        """
        result = []
        splitted_intervals = [tuple(interv.split(rounding)) for interv in self.intervals]
        for i in range(0, pow(2, self.dimension()), 1):
            num_bits = self.dimension()
            bits = [(i >> bit) & 1 for bit in range(num_bits - 1, -1, -1)]
            result.append(HyperRectangle(*[splitted_intervals[i][x] for i, x in zip(range(0, self.dimension()), bits)]))
        return result

    def split_in_single_dimension(self, dimension, rounding: int):
        intervals_1 = [(interval.split(rounding)[0] if ind == dimension else interval) for ind, interval in enumerate(self.intervals)]
        intervals_2 = [(interval.split(rounding)[1] if ind == dimension else interval) for ind, interval in enumerate(self.intervals)]
        return [HyperRectangle(*intervals_1), HyperRectangle(*intervals_2)]

    def split(self, rounding: int):
        diff = [x.width() for x in self.intervals]
        dimension = max(range(len(diff)), key=lambda i: diff[i])
        dom1 = [(interval.split(rounding)[0] if ind == dimension else interval.copy()) for ind, interval in enumerate(self.intervals)]
        dom2 = [(interval.split(rounding)[1] if ind == dimension else interval.copy()) for ind, interval in enumerate(self.intervals)]
        sub_domains = [HyperRectangle(*dom1), HyperRectangle(*dom2)]
        return sub_domains

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

    def open(self):
        return HyperRectangle(*[i.open() for i in self.intervals])

    def open_closed(self):
        return HyperRectangle(*[i.open_closed() for i in self.intervals])

    def closed_open(self):
        return HyperRectangle(*[i.closed_open() for i in self.intervals])

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
            middle_interval = Interval(new_interval_list[0].right_bound(), new_interval_list[1].left_bound(), BoundType.negated(new_interval_list[0].right_bound_type()),
                                       BoundType.negated(new_interval_list[1].left_bound_type()))  # todo negated?
            changed_interval_list = list(self.intervals)
            changed_interval_list[dimension] = middle_interval
            hrect_list.append(HyperRectangle(*changed_interval_list))

        else:
            if len(new_interval_list) > 0:
                # the cutted box
                changed_interval_list = list(self.intervals)
                changed_interval_list[dimension] = new_interval_list[0]
                hrect_list.append(HyperRectangle(*changed_interval_list))

                # the rest which have to be cutted recursively
                changed_interval_list = list(self.intervals)
                changed_interval_list[dimension] = other.intervals[dimension]
                hrect_list.append(HyperRectangle(*changed_interval_list))  # else:  #     changed_interval_list = list(self.intervals)  #     hrect_list.append(HyperRectangle(*changed_interval_list))

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
            if len(current_rect_list) > 0:
                hrect_list.append(current_rect_list[0])
                hrect_list.extend(current_rect_list[1].setminus(other, dimension + 1))
            else:
                hrect_list.extend(self.setminus(other, dimension + 1))
        return hrect_list

    def round(self, rounding: int):
        return HyperRectangle(*[i.round(rounding) for i in self.intervals])

    def to_tuple(self):
        return tuple([(interval.left_bound(), interval.right_bound()) for interval in self.intervals])

    def to_coordinates(self):
        result = []
        for interval in self.intervals:
            result.extend([interval.left_bound(), interval.right_bound()])
        return tuple(result)

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

    def __getitem__(self, key) -> Interval:
        return self.intervals[key]

    def assign(self, action):
        return HyperRectangle_action.from_hyperrectangle(self, action)


class HyperRectangle_action(HyperRectangle):
    def __init__(self, *intervals, action=None):
        """
        :param intervals: Multiple Intervals as arguments
        """
        super().__init__(*intervals)
        self.action = action

    @classmethod
    def from_hyperrectangle(cls, hyperrectangle, action):
        return cls(*[Interval(interval.left_bound(), interval.right_bound(), interval.left_bound_type(), interval.right_bound_type()) for interval in hyperrectangle.intervals], action=action)

    def split(self, rounding: int):
        domains = super().split(rounding)
        domains = [x.assign(self.action) for x in domains]
        return domains

    def to_tuple(self):
        return super().to_tuple(), self.action

    def remove_action(self):
        return HyperRectangle(*[Interval(interval.left_bound(), interval.right_bound(), interval.left_bound_type(), interval.right_bound_type()) for interval in self.intervals])

    def __repr__(self):
        return f"({super(HyperRectangle_action, self).__repr__()}, {self.action})"

    def __hash__(self):
        return hash((self.intervals, self.action))

    def __eq__(self, other):
        return other.action == self.action and super().__eq__(other)
