import re
from enum import Enum

class BoundType(Enum):
    open = 0
    closed = 1

    def __int__(self):
        return 0 if self == BoundType.open else 1

    def negated(self):
        return BoundType.closed if self == BoundType.open else BoundType.open


# (a <= X <= b) => [a,b]
# Will be updated by pycarl constraints, incl. parser.
# This is a dict as a switch-case workaround
_rel_mapping = {"<=": BoundType.closed, "<": BoundType.open, ">=": BoundType.closed, ">": BoundType.open}


def constraint_to_interval(input, internal_parse_func):
    decimals = re.compile(r"<=|<|>=|>")
    bounds = decimals.split(input)
    relations = decimals.findall(input)
    left_bound = bounds[0]
    right_bound = bounds[2]
    try:
        left_bound = internal_parse_func(left_bound)
        right_bound = internal_parse_func(right_bound)
        assert len(relations) == 2
        return Interval(left_bound, right_bound, _rel_mapping[relations[0]], _rel_mapping[relations[1]])
    except ValueError:
        return None


def string_to_interval(input_str, internal_parse_func):
    """
    Parsing intervals
    
    :param input_str: A string of the form [l,r] or (l,r) or the like.
    :param internal_parse_func: the function to parse l and r, in case they are not -infty or infty, respectively
    :return: An interval
    """
    assert isinstance(input_str, str)
    input_str = input_str.strip()
    if input_str[0] == "(":
        left_bt = BoundType.open
    elif input_str[0] == "[":
        left_bt = BoundType.closed
    else:
        raise RuntimeError("Cannot parse the interval given by: " + input_str + ". Expected '(' or '[' at the start.")

    if input_str[-1] == ")":
        right_bt = BoundType.open
    elif input_str[-1] == "]":
        right_bt = BoundType.closed
    else:
        raise RuntimeError("Cannot parse the interval given by: " + input_str + ". Expected ')' or ']' at the end.")

    inbetween = input_str[1:-1]
    bounds = inbetween.split(",")
    if len(bounds) != 2:
        raise RuntimeError("Cannot parse the interval given by: " + input_str + ". Expected exactly one comma in the interval.")

    if bounds[0] == "-infty":
        left_value = float('-inf')
        if left_bt == BoundType.closed:
            raise ValueError("Invalid interval {}: Infinity cannot have a closed bound.".format(input_str))
    else:
        left_value = internal_parse_func(bounds[0])
    if bounds[1] == "infty":
        right_value = float('inf')
        if right_bt == BoundType.closed:
            raise ValueError("Invalid interval {}: Infinity cannot have a closed bound".format(input_str))
    else:
        right_value = internal_parse_func(bounds[1])
    return Interval(left_value, right_value, left_bt, right_bt)


def create_embedded_closed_interval(interval, epsilon):
    """
    For an (half) open interval from l to r, create a closed interval [l+eps, r-eps]. 
    
    :param interval: The interval which goes into the method
    :param epsilon: An epsilon offset used to close.
    :return: A closed interval.
    """

    if interval.is_closed():
        return interval

    if not interval.is_bounded():
        raise ValueError("Cannot create embedded closes interval, as infinity plus/minus epsilon remains infinity.")

    if interval.left_bound_type() == BoundType.open:
        if interval.right_bound_type() == BoundType.open:
            if interval.width() <= 2 * epsilon:
                raise ValueError("Interval is not large enough.")
            return Interval(interval.left_bound() + epsilon, interval.right_bound() - epsilon, BoundType.closed, BoundType.closed)

    if interval.width() <= epsilon:
        raise ValueError("Interval is not large enough.")

    if interval.left_bound_type == BoundType.open:
        return Interval(interval.left_bound + epsilon, interval.right_bound(), BoundType.closed, BoundType.closed)
    return Interval(interval.left_bound, interval.right_bound() - epsilon, BoundType.closed, BoundType.closed)


class Interval:
    """
    Interval class for arbitrary (constant) types.
    Construction from string is possible via string_to_interval
    """

    def __init__(self, left_value, right_value, left_bt=BoundType.closed, right_bt=BoundType.open):
        """
        Construct an interval
        
        :param left_value: The lower bound, or -pycarl.inf if unbounded from below
        :param left_bt: If the lower bound is open or closed
        :type left_bt: BoundType
        :param right_value: The upper bound, or or pycarl.inf if unbounded from below
        :param right_bt: If the upper bound is open or closed
        :type right_bt: BoundType
        """
        self._left_bound_type = left_bt
        self._left_value = left_value
        self._right_bound_type = right_bt
        self._right_value = right_value
        assert left_value != float('inf') or left_bt == BoundType.open
        assert right_value is not None or right_bt == BoundType.open

    def left_bound(self):
        return self._left_value

    def right_bound(self):
        return self._right_value

    def left_bound_type(self):
        return self._left_bound_type

    def right_bound_type(self):
        return self._right_bound_type

    def empty(self):
        """
        Does the interval contain any points.
        
        :return: True, iff there exists a point in the interval.
        """
        if self._left_value == self._right_value:
            return self._left_bound_type == BoundType.open or self._right_bound_type == BoundType.open
        return self._left_value > self._right_value

    def contains(self, pt):
        """
        Does the interval contain a specific point
        
        :param pt: A value
        :return: True if the value lies between the bounds.
        """
        if self._left_value is None and self._right_value is None: return True
        if self._left_value is None and pt < self._right_value: return True
        if self._right_value is None and pt > self._left_value: return True
        if self._left_value < pt < self._right_value: return True
        if pt == self._left_value and self._left_bound_type == BoundType.closed: return True
        if pt == self._right_value and self._right_bound_type == BoundType.closed: return True
        return False

    def is_closed(self):
        """
        Does the interval have closed bounds on both sides.
        
        :return: True iff both bounds are closed.
        """
        return self.right_bound_type() == BoundType.closed and self.left_bound_type() == BoundType.closed

    def width(self):
        """
        The width of the interval
        
        :return: right bound - left bound, if bounded from both sides, and math.inf otherwise
        """
        return self._right_value - self._left_value

    def center(self):
        """
        Gets the center of the interval. 
        
        :return: 
        """
        return (self._right_value + self._left_value) / 2

    def split(self, precision: int):
        """
        Split the interval in two equally large halfs. Can only be called on bounded intervals
        
        :return: Two intervals, the first from the former left bound to (excluding) middle point (leftbound + rightbound)/2, 
                                the second from the middle point (including) till the former right bound
        """
        mid = round(self._left_value + self.width() / 2, precision)
        return Interval(self._left_value, mid, self._left_bound_type, BoundType.open), Interval(mid, self._right_value, BoundType.closed, self._right_bound_type)

    def close(self):
        """
        Create an interval with all bounds closed. Can not be called on unbounded intervals
        
        :return: A new interval which has closed bounds instead.
        """
        assert self._left_value != float('-inf') and self._right_value != float('inf')
        return Interval(self._left_value, self._right_value, BoundType.closed, BoundType.closed)

    def open(self):
        """
        Create an interval with all bounds open. 

        :return: A new interval which has open bounds instead
        """
        return Interval(self._left_value, self._right_value, BoundType.open, BoundType.open)

    def open_closed(self):
        """
        Create an interval with all bounds open.

        :return: A new interval which has open closed bounds instead
        """
        return Interval(self._left_value, self._right_value, BoundType.open, BoundType.closed)

    def closed_open(self):
        """
        Create an interval with all bounds open.

        :return: A new interval which has closed open bounds instead
        """
        return Interval(self._left_value, self._right_value, BoundType.closed, BoundType.open)

    def intersect(self, other):
        """
        Compute intersection between to intervals
        
        :param other: 
        :return: 
        """
        assert isinstance(other, Interval)

        # create new left bound
        if self._left_value > other._left_value:
            newleft = self._left_value
            newLB = self._left_bound_type
        elif self._left_value < other._left_value:
            newleft = other._left_value
            newLB = other._left_bound_type
        else:
            newleft = self._left_value
            newLB = self._left_bound_type if self._left_bound_type == BoundType.open else other._left_bound_type

        # create new right bound
        if self._right_value < other._right_value:
            newright = self._right_value
            newRB = self._right_bound_type
        elif self._right_value > other._right_value:
            newright = other._right_value
            newRB = other._right_bound_type
        else:
            newright = self._right_value
            newRB = self._right_bound_type if self._right_bound_type == BoundType.open else other._right_bound_type

        # what if the intersection is empty?
        return Interval(newleft, newright, newLB, newRB)

    def __str__(self):
        return ("(" if self._left_bound_type == BoundType.open else "[") + str(self._left_value) + "," + str(self._right_value) + (")" if self._right_bound_type == BoundType.open else "]")

    def __repr__(self):
        return ("(" if self._left_bound_type == BoundType.open else "[") + repr(self._left_value) + "," + repr(self._right_value) + (")" if self._right_bound_type == BoundType.open else "]")

    def __eq__(self, other):
        assert isinstance(other, Interval)
        if self.empty() and other.empty():
            return True
        if not self._left_bound_type == other.left_bound_type():
            return False
        if not self._left_value == other.left_bound():
            return False
        if not self._right_bound_type == other.right_bound_type():
            return False
        if not self._right_value == other.right_bound():
            return False
        return True

    def __hash__(self):
        if self.empty():
            return 0
        return hash(self._left_value) ^ hash(self._right_value) + int(self._left_bound_type) + int(self._right_bound_type)

    def round(self, rounding: int):
        return Interval(round(self._left_value, rounding), round(self._right_value, rounding), self._left_bound_type, self._right_bound_type)

    def setminus(self, other):
        """
        Compute the setminus of two rectangles
        
        :param other: 
        :return: 
        """
        intersectionInterval = self.intersect(other)
        if intersectionInterval.empty():
            return [self]
        elif intersectionInterval == self:
            return []
        else:
            if self._left_value == intersectionInterval._left_value:
                if self._left_value == intersectionInterval._right_value:
                    return [Interval(self._left_value, self._right_value, BoundType.negated(intersectionInterval._right_bound_type), self._right_bound_type)]
                else:
                    if intersectionInterval._left_bound_type == BoundType.open and self._left_bound_type == BoundType.closed:
                        return [Interval(self._left_value, self._left_value, self._left_bound_type, self._left_bound_type),
                                Interval(intersectionInterval._right_value, self._right_value, BoundType.negated(intersectionInterval._right_bound_type), self._right_bound_type)]
                    else:
                        return [Interval(intersectionInterval._right_value, self._right_value, BoundType.negated(intersectionInterval._right_bound_type), self._right_bound_type)]
            elif self._right_value == intersectionInterval._right_value:
                if self._right_value == intersectionInterval._left_value:
                    return [Interval(self._left_value, self._right_value, self._left_bound_type, BoundType.negated(intersectionInterval._right_bound_type))]
                else:
                    if intersectionInterval._right_bound_type == BoundType.open and self._right_bound_type == BoundType.closed:
                        return [Interval(self._right_value, self._right_value, self._right_bound_type, self._right_bound_type),
                                Interval(self._left_value, intersectionInterval._left_value, self._left_bound_type, BoundType.negated(intersectionInterval._left_bound_type))]
                    else:
                        return [Interval(self._left_value, intersectionInterval._left_value, self._left_bound_type, BoundType.negated(intersectionInterval._left_bound_type))]
            else:
                return [Interval(self._left_value, intersectionInterval._left_value, self._left_bound_type, BoundType.negated(intersectionInterval._left_bound_type)),
                        Interval(intersectionInterval._right_value, self._right_value, BoundType.negated(intersectionInterval._right_bound_type), self._right_bound_type)]

    def copy(self):
        return Interval(self._left_value, self._right_value, self._left_bound_type, self._right_bound_type)
