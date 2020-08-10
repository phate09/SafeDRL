# cython: profile=False

cdef class Interval:
    """
    Interval class for arbitrary (constant) types.
    Construction from string is possible via string_to_interval
    """
    cdef bint _is_left_bound_closed, _is_right_bound_closed
    cdef double _left_value, _right_value
    def __init__(self, double left_value, double right_value, bint left_closed=True, bint right_closed=False):
        """
        Construct an interval
        
        :param left_value: The lower bound, or -pycarl.inf if unbounded from below
        :param left_closed: If the lower bound is open or closed
        :type left_closed: BoundType
        :param right_value: The upper bound, or or pycarl.inf if unbounded from below
        :param right_closed: If the upper bound is open or closed
        :type right_closed: BoundType
        """
        self._is_left_bound_closed = left_closed
        self._left_value = left_value
        self._is_right_bound_closed = right_closed
        self._right_value = right_value
        assert left_value != float('inf') or left_closed == False
        assert right_value is not None or right_closed == False

    cpdef double left_bound(self):
        return self._left_value

    cpdef double right_bound(self):
        return self._right_value

    cpdef bint left_bound_closed(self):
        return self._is_left_bound_closed

    cpdef bint right_bound_closed(self):
        return self._is_right_bound_closed

    cpdef bint empty(self):
        """
        Does the interval contain any points.
        
        :return: True, iff there exists a point in the interval.
        """
        if self._left_value == self._right_value:
            return self._is_left_bound_closed == False or self._is_right_bound_closed == False
        return self._left_value > self._right_value

    cpdef bint contains(self, double pt):
        """
        Does the interval contain a specific point
        
        :param pt: A value
        :return: True if the value lies between the bounds.
        """
        if self._left_value is None and self._right_value is None: return True
        if self._left_value is None and pt < self._right_value: return True
        if self._right_value is None and pt > self._left_value: return True
        if self._left_value < pt < self._right_value: return True
        if pt == self._left_value and self._is_left_bound_closed == True: return True
        if pt == self._right_value and self._is_right_bound_closed == True: return True
        return False

    cpdef bint is_closed(self):
        """
        Does the interval have closed bounds on both sides.
        
        :return: True iff both bounds are closed.
        """
        return self.right_bound_closed() == True and self.left_bound_closed() == True

    cpdef double width(self):
        """
        The width of the interval
        
        :return: right bound - left bound, if bounded from both sides, and math.inf otherwise
        """
        return self._right_value - self._left_value

    cpdef double center(self):
        """
        Gets the center of the interval. 
        
        :return: 
        """
        return (self._right_value + self._left_value) / 2

    cpdef tuple split(self, int precision):
        """
        Split the interval in two equally large halfs. Can only be called on bounded intervals
        
        :return: Two intervals, the first from the former left bound to (excluding) middle point (leftbound + rightbound)/2, 
                                the second from the middle point (including) till the former right bound
        """
        mid = round(self._left_value + self.width() / 2, precision)
        return Interval(self._left_value, mid, self._is_left_bound_closed, False), Interval(mid, self._right_value, True, self._is_right_bound_closed)

    cpdef Interval close(self):
        """
        Create an interval with all bounds closed. Can not be called on unbounded intervals
        
        :return: A new interval which has closed bounds instead.
        """
        assert self._left_value != float('-inf') and self._right_value != float('inf')
        return Interval(self._left_value, self._right_value, True, True)

    cpdef Interval open(self):
        """
        Create an interval with all bounds open. 

        :return: A new interval which has open bounds instead
        """
        return Interval(self._left_value, self._right_value, False, False)

    cpdef Interval open_closed(self):
        """
        Create an interval with all bounds open.

        :return: A new interval which has open closed bounds instead
        """
        return Interval(self._left_value, self._right_value, False, True)

    cpdef Interval closed_open(self):
        """
        Create an interval with all bounds open.

        :return: A new interval which has closed open bounds instead
        """
        return Interval(self._left_value, self._right_value, True, False)

    cpdef Interval intersect(self, Interval other):
        """
        Compute intersection between to intervals
        
        :param other: 
        :return: 
        """
        assert isinstance(other, Interval)

        # create new left bound
        if self._left_value > other._left_value:
            newleft = self._left_value
            newLB = self._is_left_bound_closed
        elif self._left_value < other._left_value:
            newleft = other._left_value
            newLB = other._is_left_bound_closed
        else:
            newleft = self._left_value
            newLB = self._is_left_bound_closed if self._is_left_bound_closed == False else other._is_left_bound_closed

        # create new right bound
        if self._right_value < other._right_value:
            newright = self._right_value
            newRB = self._is_right_bound_closed
        elif self._right_value > other._right_value:
            newright = other._right_value
            newRB = other._is_right_bound_closed
        else:
            newright = self._right_value
            newRB = self._is_right_bound_closed if self._is_right_bound_closed == False else other._is_right_bound_closed

        # what if the intersection is empty?
        return Interval(newleft, newright, newLB, newRB)

    def __str__(self):
        return ("(" if self._is_left_bound_closed == False else "[") + str(self._left_value) + "," + str(self._right_value) + (")" if self._is_right_bound_closed == False else "]")

    def __repr__(self):
        return ("(" if self._is_left_bound_closed == False else "[") + repr(self._left_value) + "," + repr(self._right_value) + (")" if self._is_right_bound_closed == False else "]")

    def __eq__(self, other):
        assert isinstance(other, Interval)
        if self.empty() and other.empty():
            return True
        if not self._is_left_bound_closed == other.left_bound_closed():
            return False
        if not self._left_value == other.left_bound():
            return False
        if not self._is_right_bound_closed == other.right_bound_closed():
            return False
        if not self._right_value == other.right_bound():
            return False
        return True

    def __hash__(self):
        if self.empty():
            return 0
        return hash(self._left_value) ^ hash(self._right_value) + int(self._is_left_bound_closed) + int(self._is_right_bound_closed)

    cpdef Interval round(self, int rounding):
        return Interval(round(self._left_value, rounding), round(self._right_value, rounding), self._is_left_bound_closed, self._is_right_bound_closed)

    def setminus(self, Interval other):
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
                    return [Interval(self._left_value, self._right_value, not (intersectionInterval._is_right_bound_closed), self._is_right_bound_closed)]
                else:
                    if intersectionInterval._is_left_bound_closed == False and self._is_left_bound_closed == True:
                        return [Interval(self._left_value, self._left_value, self._is_left_bound_closed, self._is_left_bound_closed),
                                Interval(intersectionInterval._right_value, self._right_value, not (intersectionInterval._is_right_bound_closed), self._is_right_bound_closed)]
                    else:
                        return [Interval(intersectionInterval._right_value, self._right_value, not (intersectionInterval._is_right_bound_closed), self._is_right_bound_closed)]
            elif self._right_value == intersectionInterval._right_value:
                if self._right_value == intersectionInterval._left_value:
                    return [Interval(self._left_value, self._right_value, self._is_left_bound_closed, not (intersectionInterval._is_right_bound_closed))]
                else:
                    if intersectionInterval._is_right_bound_closed == False and self._is_right_bound_closed == True:
                        return [Interval(self._right_value, self._right_value, self._is_right_bound_closed, self._is_right_bound_closed),
                                Interval(self._left_value, intersectionInterval._left_value, self._is_left_bound_closed, not (intersectionInterval._is_left_bound_closed))]
                    else:
                        return [Interval(self._left_value, intersectionInterval._left_value, self._is_left_bound_closed, not (intersectionInterval._is_left_bound_closed))]
            else:
                return [Interval(self._left_value, intersectionInterval._left_value, self._is_left_bound_closed, not (intersectionInterval._is_left_bound_closed)),
                        Interval(intersectionInterval._right_value, self._right_value, not (intersectionInterval._is_right_bound_closed), self._is_right_bound_closed)]

    cpdef Interval copy(self):
        return Interval(self._left_value, self._right_value, self._is_left_bound_closed, self._is_right_bound_closed)
