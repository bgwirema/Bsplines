import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import sparse
import Spline



def penaltyMatrix(order, knots, sparse=False, BSpline = True, positive = False, der=2):
    """
        Evaluate penalty matrix for B or N splines
        L2 gram matrix of the second derivatives of the splines

        Parameters
        ----------
        order : int
            Order of the splines (polynomial degree + 1)
        knots : array-like
            Knots without duplicate padding
        sparse : bool
            (Optional, default False) Return the matrix in dense or sparse
        BSpline : bool
            (Optional, default True) Calculate penalty matrix for B or N splines
        positive : bool
            (Optional, default False) Normalize the N splines to be positive.
            Only considered if BSpline is False
        der: int
            (Optional, default 2) Derivative by which to calculate the penalty matrix


        Returns
        -------
        penaltyMatrix : ndarray or scipy.sparse.dia_array
            rows and columns correspond to the spline basis indexes

            ndarray if sparse is False
            scipy.sparse.dia_array if sparse is True

        Notes
        -----
        Algorithm:
            Let G be the L2 gram matrix of B-splines of order - der, and let D be the linear transformation matrix from
            B-splines of order - der to the derivatives of B (or N) - splines of order.
            The penalty matrix equals  D^T G D

            This is calculated sparsely and then converted to dense formate if desired.

            Time complexity:

            Space complexity:
    """

    # get the sparse gram matrix for order-derivative
    gramMatrix = BGram(order - der, knots, sparse= True)
    # get the sparse derivative matrix for order, taking into account BSpline and positive
    derivativeMatrix = BDerivativeMultiples(order, knots, der, sparse= True, BSpline= BSpline, positive= positive)
    # combine
    penaltyMatrix = derivativeMatrix.T @ gramMatrix @ derivativeMatrix

    if sparse:
        return penaltyMatrix
    else:
        return penaltyMatrix.todense()


def BDerivativeMultiples(order, knots, der, sparse=False, diagonal = False, BSpline = True, positive = False):
    """
        Evaluate the linear multiples needed to construct derivatives of B or N - splines
        Evaluates the matrix representing the linear transformation from B-splines of order - der to B or N splines of
        order

        Parameters
        ----------
        order : int
            Order of the splines (polynomial degree + 1)
        knots : array-like
            Knots without duplicate padding
        der: int
            Derivative to evaluate
        sparse : bool
            (Optional, default False) Return the matrix in dense or sparse
        diagonal : bool
            (Optional, default False) Return the diagonals in an ndarray format
            (Takes priority over sparse)
        BSpline : bool
            (Optional, default True) Compute the derivative multiples for B-splines or N-splines
        positive : bool
            (Optional, default False) Normalize the N splines to be positive.
            Only considered if BSpline is False


        Returns
        -------
        derivativeMultiples : ndarray or scipy.sparse.dia_array
            rows correspond to the B-Spline basis indexes for order-der
            rows correspond to the der'th derivatives of B (or N)-Spline basis indexes for order

            ndarray if sparse is False
            scipy.sparse.dia_array if sparse is True

        Notes
        -----
        Algorithm:
            Working over the diagonals of the matrix, using dynamic programming, build up the matrix according to the
            recursive relation:

            B'_{index,order} = B_{index-1,order-1} / (knots[index+order-1] -
                knots[index]) - B_{index,order-1} / (knots[index + order] - knots[index+1])

            Then converted to dense formate if desired.

            Time complexity:
                if sparse: ''O(der^2 * (order + len(knots)))''
                else: ''O(der^2 * (order + len(knots)) + (order + len(knots))^2)''

            Space complexity:
                if sparse: ''O(der * (order + len(knots)))''
                else: ''O((order + len(knots))^2)''
    """

    # compute the second derivatives for the end points needed for N splines
    if not BSpline:
        s0,s1,s2 = SecondDerivatives(order, knots, True)
        s_3,s_2,s_1 = SecondDerivatives(order, knots, False)

    # initialize useful variables and pad the knots
    degree = order - 1
    dim = (degree + len(knots) - 1)
    knots = np.pad(knots, degree, 'edge')

    # initialize the diagonals to be the identity matrix corresponding to der = 0
    diag = np.zeros((dim, der + 1))
    diag[der:, -1] = 1

    # build up the derivative multiply matrix
    # after each iteration diag[:,-d-1:] stores the information for a derivative multiple matrix for the dth derivatives
    left = knots[:dim]
    deg = degree - der  # polynomial degree tracker
    for d in range(der):
        # nonzero locations of the array
        rows = np.arange(der - d, dim).reshape(-1, 1)
        cols = np.arange(der - d, der + 1)

        # update deg and the right end points of spline supports
        deg = deg + 1
        right = knots[rows + deg]

        # actual derivative formula here:
        diag[rows, cols] = diag[rows, cols] * deg / (right - left[rows])
        diag[rows - 1, cols - 1] -= diag[rows, cols]


    # cut of the zero padding used in calculation
    diagReshaped = np.zeros((dim - der, der + 1))
    for i in range(der + 1):
        # diagonals are read left to right, sliding window 1 down each time
        diagReshaped[:, i] = diag[i:dim - der + i, i]

    # alter into N-splines if desired
    # working with diagonals apply the transformation of B -> N splines on the columns
    if not BSpline:
        # normalize the left most endpoint to have 2nd derivative 0
        diagReshaped[0, 1] -= s1 / s0 * diagReshaped[0, 0]
        diagReshaped[0, 2] -= s2 / s0 * diagReshaped[0, 0]
        diagReshaped[0, 0] = 0

        # normalize the right most endpoint to have 2nd derivative 0
        diagReshaped[-1, -2] -= s_2 / s_1 * diagReshaped[-1, -1]
        diagReshaped[-1, -3] -= s_3 / s_1 * diagReshaped[-1, -1]
        diagReshaped[-1, -1] = 0

        # normalize the N splines to be positive if desired
        if positive:
            diagReshaped[0, 2] -= s2 / s1 * diagReshaped[0, 1]
            diagReshaped[-1, -3] -= s_3 / s_2 * diagReshaped[-1, -2]


    # reshape into matrices

    if diagonal:
        return diagReshaped
    else:
        # put into sparse matrices by diagonals
        if BSpline:
            DMat = scipy.sparse.dia_array((dim - der, dim))
            for i in range(der + 1):
                DMat.setdiag(diagReshaped[:,i], i)
        else:
            # if N-splines then the there are 2 fewer columns
            DMat = scipy.sparse.dia_array((dim - der, dim-2))

            # first and last diagonal have 1 fewer elements
            DMat.setdiag(diagReshaped[1:-1, 0], -1)
            for i in range(1,der):
                DMat.setdiag(diagReshaped[:,i], i-1)
            DMat.setdiag(diagReshaped[:-1, -1], der-1)

        if sparse:
            return DMat
        else:
            return DMat.todense()


def SecondDerivatives(order, knots, first = True):
    """
        Evaluates the second derivatives of the first 3 B-splines at the first or last knot

        Parameters
        ----------
        order : int
            Order of the splines (polynomial degree + 1)
        knots : array-like
            Knots without duplicate padding
        first : bool
            (default = True) evaluate at the first knot or last knot

        Returns
        -------
        if first:
            s0,s1,s2 : float
                the second derivatives at the first knot for splines with index 0,1,2
        else:
            s_2,s_2,s_1 : float
                the second derivatives at the last knot for splines with index -1,-2,-3

        Notes
        -----
        Algorithm:
            A simplified version of the recursive relation:

            B'_{index,order} = B_{index-1,order-1} / (knots[index+order-1] -
                knots[index]) - B_{index,order-1} / (knots[index + order] - knots[index+1])

            knowing that at the first knot only the first B-spline = and the rest equal 0, and similar for the last
            knot and the last spline.

        Time complexity: ''O(1)''

        Space complexity: ''O(1)''
    """

    degree = order -1
    knots = np.pad(knots, degree, 'edge')
    degFac = degree * (degree - 1)
    if first:
        width1 = knots[order] - knots[0]
        width2 = knots[order + 1] - knots[1]
        s0 = degFac / width1 ** 2
        s1 = -degFac / width1 * (1 / width1 + 1 / width2)
        s2 = degFac / width1 / width2
        return s0,s1,s2
    else:
        width1 = knots[-1 - order] - knots[-1]
        width2 = knots[-2 - order] - knots[-2]
        s_3 = degFac / width1 / width2
        s_2 = -degFac / width1 * (1 / width1 + 1 / width2)
        s_1 = degFac / width1 ** 2
        return s_3, s_2, s_1


def BGram(order, knots, sparse = False, diagonal = False):
    """
        Evaluate the L2 gram matrix for the B-spline basis

        Parameters
        ----------
        order : int
            Order of the splines (polynomial degree + 1)
        knots : array-like
            Knots without duplicate padding
        sparse : bool
            (Optional, default False) Return the matrix in dense or sparse
        diagonal : bool
            (Optional, default False) Return the diagonals in an ndarray format
            (Takes priority over sparse)

        Returns
        -------
        gramMatrix : ndarray or scipy.sparse.dia_array
            rows and columns correspond to the B-Spline basis indexes for order

            ndarray if sparse is False
            scipy.sparse.dia_array if sparse is True

        Notes
        -----
        Algorithm:
            if order == 1:
                The splines are constant equal to 1. The gram matrix is the diagonal matrix of the interval widths
            if order == 2:
                The splines are linear and the integral of their products can be simplified using u-sub down to simple
                multiples of the widths of the intervals.
            if order > 2:
                The partial truncated polynomial coefficients are calculated using BCoeffs.
                Then the L2 gram matrix in this basis is computed using TruncInnerProduct to locally compute the
                integrals in a sliding window fashion over the supports of the splines.
                During calculation the change of basis to B-splines is computed using the matrix returned by BCoeffs.

                Theoretically:
                Let B be the linear transformation from the partial truncated polynomial basis to the B-spline basis
                Let G be the L2 gram matrix on the partial truncated polynomial basis

                The L2 gram matrix on the B-spline basis equals B^T G B

            The computation is done sparsely on the upper diagonals of the gram matrix.
            It is then converted to dense format if desired.


        Time complexity:
            if order<=2:
                ''O(order * (order + len(knots)))''
            elif sparse:
                ''O(order^4 + order^3 * len(knots))''
            else:
                'O(order^4 + order^3 * len(knots) + len(knots)^2)''

        Space complexity:
            if order<=2:
                ''O(order * (order + len(knots)))''
            elif sparse:
                ''O(order^2 + order*knots)''
            else:
                ''O(order^2 + order*knots + len(knots)^2)''
    """

    # compute useful variables
    degree = order - 1
    dim = (degree + len(knots) - 1)
    knots = np.asarray(knots)

    if degree == 0:
        # gram matrx is the identity matrix
        width = knots[1:] - knots[:-1]
        diag = np.reshape(width,(-1,1))
    elif degree == 1:
        # linear splines
        diag = np.zeros((dim, order))

        # since the linear B-splines all have peaks at y=1 their product integrals can be simplified heavily using
        # u-sub to be combinations of the widths of the intervals
        width = knots[1:] - knots[:-1]

        diag[:-1,0] += width / 3
        diag[1:, 0] += width / 3
        diag[:-1, 1] = width / 6
    elif degree > 1:
        # non linear splines

        # compute the change of basis matrix from the partial truncated polynomial basis to the B-spline basis
        bTang = BCoeffs(order, knots, diagonal=True)  # O(degree^2 * (degree+k))

        # add padding to the knots
        knots = np.pad(knots, degree, 'edge')

        # init the diagonals of the gram matrix to 0
        diag = np.zeros((dim, order))

        # sliding window of the L2 gram matrix on the partial truncated polynomial basis
        slidingGram = np.zeros((order + 1, order + 1))  # has 0 padding for cleanliness later

        # function for local computation of the inner products
        shift = lambda bIndex1, bIndex2, kIndex: TruncInnerProduct(degree, knots, bIndex1, bIndex2, kIndex)

        for i in range(dim):
            # locally calculate the inner product integrals for the truncated polynomial basis
            # locally: only over the support of B-spline_i
            # shift: add on the integral over the next interval
            for j in range(min(order, dim - i)):
                for k in range(j, min(order, dim - i)):
                    slidingGram[j, k] = slidingGram[j + 1, k + 1] + shift(i + j, i + k, i + degree)  # O(order)


            # compute part of B^T G B
            # using the sliding gram, only over the splines that will have non zeros inner products

            # left multiplication
            leftVec = np.zeros(order)
            for j in range(order):
                for k in range(order):
                    leftVec[j] += slidingGram[min(j, k), max(j, k)] * bTang[i, k]

            # right multiplication (now it's a dot product)
            for j in range(min(order, dim - i)):
                for k in range(min(order, dim - i) - j):
                    diag[i, j] += leftVec[j + k] * bTang[i + j, k]
    else:
        # shouldn't run
        assert order >= 1  # will fail


    if diagonal:
        return diag

    # put into to a sparse matrix
    GMat = scipy.sparse.dia_array((dim, dim))
    for i in range(order):
        GMat.setdiag(diag[:dim-i, i], i) # cut off lower triangle of zeros
        if i > 0:
            GMat.setdiag(diag[:dim-i, i], -i)

    if sparse:
        return GMat
    else:
        return GMat.todense()


def BCoeffs(order, knots, sparse = False, diagonal = False):
    """
        Evaluate a subset of the polynomial coefficients of B splines.
        All coefficients for the first piecewise interval.
        Only the maximal degree coefficient for each of the next intervals

        Evaluate the basis change matrix from the partial truncated polynomial basis to the B-spline basis

        Parameters
        ----------
        order : int
            Order of the splines (polynomial degree + 1)
        knots : array-like
            Knots without duplicate padding
        sparse : bool
            (Optional, default False) Return the matrix in dense or sparse
        diagonal : bool
            (Optional, default False) Return the diagonals in an ndarray format
            (Takes priority over sparse)

        Returns
        -------
        BMatrix : ndarray or scipy.sparse.dia_array
            rows correspond to the B-Spline basis indexes for order
            cols corresponding to the coefficients of the partial truncated polynomials

            ndarray if sparse is False
            scipy.sparse.dia_array if sparse is True

        Notes
        -----
        Algorithm:
            Working over the diagonals of the matrix, build up the B-spline coefficients using a dynamic programming
            implementation of the cox-de boor algorithm modified to the partial truncated polynomial basis.

            The computation is done sparsely and then converted to dense format if desired.

        Time complexity:
            if sparse:
                ''O(degree^2 * (degree + len(knots)))''
            else:
                ''O(degree^2 * (degree + len(knots) + (degree + len(knots)^2))''

        Space complexity:
            if sparse:
                ''O(degree * (degree + len(knots)))''
            else:
                ''O(degree * (degree + len(knots) + (degree + len(knots)^2))''
    """

    # compute useful variables and add padding to the knots
    degree = order - 1
    dim = (degree + len(knots) - 1)
    knots = np.asarray(knots)-knots[0]
    knots = np.pad(knots, degree, 'edge')

    # initialize the diagonals to be B-splines of order 1
    diag = np.zeros((dim, order))
    diag[degree:, 0] = 1


    # build up the order of the B-splines
    # after each iteration diag[order - deg-1:dim, :deg+1] stores the information for B-splines of degree deg
    left = knots[:dim]
    for deg in range(1, order):
        # the indexes of the non-zero entries in diag
        cols = np.arange(deg)
        rows = np.arange(order - deg, dim).reshape(-1, 1)

        # the right endpoints of the current supports and the width of the current supports
        right = knots[rows + deg]
        width = right - left[rows]

        # a copy of the current B-splines divided by the width of their supports
        past = diag[rows, cols] / width

        # in the cox-de boor formula for this basis multiplication by (x-knots[index]) can be broken apart since the
        # multiplication by -knots[index] only affects the coefficients of not maximal degree, stored for the first
        # interval

        # the left term in the cox-de boor algorithm
        diag[rows, cols] = past  # the multiplication by x on all coefficients
        # loop over the few non-maximal coefficients
        for i in range(1, deg):
            s = degree - i  # spline index
            for c in range(i):  # coefficient index
                diag[s, c] -= past[deg - i, c + 1] * left[s]  # the multiplication by -knots[index]

        # The right term in the cox-de boor algorithm
        diag[rows - 1, cols + 1] -= past  # the multiplication by -x on all coefficients
        # loop over the few non-maximal coefficients
        for i in range(1, deg + 1):
            s = degree - i  # spline index
            for c in range(i):  # coefficient index
                diag[s, c] += past[deg - i, c] * right[deg - i]  # the multiplication by knots[index+deg]


    if diagonal:
        return diag

    # put into a sparse matrix
    BMat = scipy.sparse.dia_array((dim, dim))
    for i in range(order):
        BMat.setdiag(diag[:dim - i, i], i)  # cut off triangle of zeros at bottom

    if sparse:
        return BMat
    else:
        return BMat.todense()


def TruncInnerProduct(degree, knots, i, j, k):
    """
        Local L2 inner product of the partial truncated polynomial basis elements
        <e_i,e_j> evaluated only on the interval [knots[k],knots[k+1]]

        Parameters
        ----------
        degree : int
            Degree of the spline basis
        knots : array-like
            Knots with duplicate padding
        i : int
            Index of the first basis element
        j : int
            Index of the second basis element
        k : int
            Knot index of the left end point of the evaluation interval

        Returns
        -------
        I : float
            The inner product of e_i and e_j evaluated on the interval [knots[k],knots[k+1]]

        Notes
        -----
        The basis elements are monomials with different center points or differences of these monomials.

        A conditional structure determines what the product is and then MonomialProductIntegral is called to evaluate
        the integrals for products of these monomial

        Time complexity: ''O(degree)''

        Space complexity: ''O(degree)''
    """

    # make i <= j
    if not i<=j:
        temp = i
        i = j
        j = temp

    # integral function with consistent parameters already specified
    mon = lambda index1, p1, index2, p2: MonomialProductIntegral(knots[index1], p1, knots[index2], p2, knots[k],
                                                                 knots[k + 1])

    # switch structure for the different basis elements
    if j < i:  # outside of the supports
        return 0
    elif i < degree:  # e_i is not a maximal degree coefficient
        if j < degree:  # e_j is also not a maximal degree coefficient
            return mon(i, i, j, j)
        elif k == j:  # e_j is maximal but evaluated on its first interval
            return mon(i, i, j, degree)
        else:  # e_j is maximal but evaluated after its first interval
            return mon(i, i, j, degree) - mon(i, i, j + 1, degree)
    else:  # e_i is maximal
        if k == j:  # e_j is evaluated on its first interval
            if k == i:  # e_i is also evaluated on its first interval
                return mon(i, degree, j, degree)
            else:  # e_i is not evaluated on its first interval
                return mon(i, degree, j, degree) - mon(i + 1, degree, j, degree)
        else: # neither e_i nor e_j are evaluated on their first intervals
            return mon(i, degree, j, degree) - mon(i + 1, degree, j, degree) - mon(i, degree, j + 1, degree) + mon(
                i + 1, degree, j + 1, degree)


def MonomialProductIntegral(x1, p1, x2, p2, low, up):
    """
        Integrates (x-x1)^p1 * (x-x2)^p2  on the interval [low,up]

        Parameters
        ----------
        x1 : float
            center of the first monomial
        p1 : int
            power of the first monomial
        x2 : float
            center of the second monomial
        p2 : int
            power of the second monomial
        low : float
            lower bound of the integral
        up : float
            upper bound of the integral

        Returns
        -------
        I : float
            The integral of (x-x1)^p1 * (x-x2)^p2  on the interval [low,up]

        Notes
        -----
        Evaluates it analytically using a taylor polynomial style change of center

        Time complexity: ''O(min(p1,p2))''

        Space complexity: ''O(min(p1,p2))''
    """

    if not p1 <= p2:  # make p1 < p2
        temp = x1
        x1 = x2
        x2 = temp
        temp = p1
        p1 = p2
        p2 = temp
    if not x1 == 0:  # make x1 == 0
        low = low - x1
        up = up - x1
        x2 = x2 - x1
    if x2 == 0:
        return (up ** (p1 + p2 + 1) - low ** (p1 + p2 + 1)) / (p1 + p2 + 1)

    # precalculate powers of x2 (used in reverse order)
    x2Pow = np.ones(p2 + 1)
    for i in range(1, p2 + 1):
        x2Pow[i] = x2Pow[i - 1] * x2

    # initialize product variables
    mul = (-1) ** (p2 % 2)  # (-1)^x * derivative product / factorial
    lowPow = low ** p1
    upPow = up ** p1

    integral = 0
    for i in range(p2 + 1):
        # update product variables
        if i > 0:
            mul *= -(p2 - i + 1) / i
        lowPow *= low
        upPow *= up
        # sum integral by polynomial term
        integral += mul * x2Pow[-1 - i] * (upPow - lowPow) / (i + p1 + 1)

    return integral