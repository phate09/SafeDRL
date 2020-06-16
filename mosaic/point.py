import math
# from prophesy.data.nice_approximation import FixedDenomFloatApproximation


def _sqrt_approx(i):
    """
    :param i: a positive number
    :return: the approximate square root of i
    """
    orig_type = type(i)
    if not isinstance(i, (int, float)):
        # if i is a rational number for which the sqrt might be not a
        # rational number, then use some approx.
        i = float(i)

    # TODO this might be much closer than necessary here.
    float_sqrt = math.sqrt(i)
    return orig_type(float_sqrt)


class Point:
    """
    An n-dimensional point class.
    """
    def __init__(self, *args):
        """
        :param args: Numerical values to represent the point. 
        """
        self.coordinates = tuple(args)

    def to_float(self):
        return Point(*[float(c) for c in self.coordinates])

    # def to_nice_rationals(self, ApproxType = FixedDenomFloatApproximation, approx_type_arg = 16384):
    #     """
    #     Transfer the coordinates into rational numbers with smaller coefficients
    #
    #     :param ApproxType: The type of approximation to use
    #     :param approx_type_arg: An argument for the approximation, typically some sort of precision of the approximation
    #     :return: A point with slightly modified coordinates
    #     :rtype: Point
    #     """
    #     approx = ApproxType(approx_type_arg)
    #     return Point(*[approx.find(c) for c in self.coordinates])

    def distance(self, other):
        """
        Computes the (Euclidean) distance between this point and another point 
        
        :param other: Another n-dimensional point
        :type other: Point
        :return: The Euclidean distance
        """
        assert self.dimension() == other.dimension()
        res = 0.0
        for i, j in zip(self.coordinates, other.coordinates):
            tres = type(res)
            res = res + tres(pow(i-j,2))
        return _sqrt_approx(res)

    def numerical_distance(self, other):
        """
        Computes the (Euclidean) distance between this point and another point, using floating point arithmetic
        
        :param other: Another n-dimensional point
        :type other: Point
        :return: 
        """
        assert self.dimension() == other.dimension()
        res = 0.0
        for i, j in zip(self.coordinates, other.coordinates):
            res = res + (pow(float(i)-float(j),2))
        return _sqrt_approx(res)

    def dimension(self):
        """
        The dimension of the point, e.g. the number of coordinates
        
        :return: The number of entries 
        """
        return len(self.coordinates)

    def projection(self, dims):
        """
        Project the point onto the selected dimensions
        
        :param dims: An iterable of dimensions to select
        :return: A len(dims)-dimensional Point
        """
        return Point(*[self.coordinates[i] for i in dims])

    def __str__(self):
        return "(" + ",".join([str(i) for i in self.coordinates]) + ")"

    def __iter__(self):
        return iter(self.coordinates)

    def __add__(self, other):
        assert self.dimension() == other.dimension()
        return Point(*[c1 + c2 for (c1, c2) in zip(self.coordinates, other.coordinates)])

    def __sub__(self, other):
        assert self.dimension() == other.dimension()
        return Point(*[c1 - c2 for (c1, c2) in zip(self.coordinates, other.coordinates)])

    def __mul__(self, sc):
        return Point(*[c1 * sc for c1 in self.coordinates])

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, key):
        return self.coordinates[key]

    def __eq__(self, other):
        return isinstance(other, Point) and self.coordinates == other.coordinates

    def __hash__(self):
        return hash(self.coordinates)

    def __repr__(self):
        return "Point({})".format(", ".join(map(repr,self.coordinates)))
