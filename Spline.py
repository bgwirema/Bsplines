import numpy as np
import matplotlib.pyplot as plt

# Evaluates B-splines
#   positive basis splines. normalized to have total sum = 1
#
# Inputs:
#   evalPoints:     list of points to evaluate the splines at   (sorted ascending)  # TODO: handle unsorted
#   deg:            degree of the splines                       (polynomial degree + 1)
#   knots:          list of knots                               (sorted ascending without duplicates)
#
# Output:
#   designMatrix:   first dim - evalPoints
#                   second dim - spline index                   (number of splines = deg + len(knots) - 2)
#
# Algorithm:
#   Modified Cox-De Boor formula
#       Bundles the splines during calculation to only run over the supports  TODO: proof
#       Algorithm adapted from matlab FDAfuns
#
# Time Complexity:
#   O(len(evalPoints) * (deg^2 + len(knots)))
#   O(res*k(d^2+k))
#   O(res*k*d^2 + res*k^2)
#
# Space Complexity:
#   O(len(evalPoints) * (deg + len(knots)))
#
# TODO: efficiency
#   has lots of overhead but a very good time complexity
#
def bSplineMatrix(evalPoints, deg, knots):
    numx = len(evalPoints)
    numk = len(knots)
    nums = deg + numk - 2

    evalPoints = np.squeeze(evalPoints)  # ensure it's a numpy array
    knots = np.concatenate(
        (np.ones(deg - 1) * knots[0], knots, np.ones(deg - 1) * knots[-1]))  # add padding to the knots

    """
    Initialize a list of the knot indexes to the left of each evaluation points
    """
    # O(numx)
    left = np.zeros(numx, int)  # knot index + padding to the left of each eval point
    index = 0
    last = evalPoints[0]
    for i in range(numx):
        x = evalPoints[i]
        assert last <= x, "evaluation points not ordered"  # TODO: add a case to handle unsorted eval points
        while not (knots[index + (deg - 1)] <= x <= knots[
            index + 1 + (deg - 1)]):  # move the index forward to find the correct knot
            index += 1
            assert index < len(knots) - 1, "evaluation points not within the knot domain"

        left[i] = index + (deg - 1)  # index + padding
        last = x

    """
    Bundled calculate the spline values
    """
    # O(numx * deg^2)
    tangled = np.zeros((deg, numx))
    tangled[0] = np.ones(numx)
    for i in range(deg - 1):  # loop up the degrees
        leftS = np.zeros(numx)
        j = 0
        for j in range(i + 1):  # loop over the tangled calculated splines
            knotR = knots[left + j + 1]  # knots right
            knotL = knots[left + (j - i)]  # knots left
            # Cox-De Boor formula
            pastS = tangled[j] / (knotR - knotL)
            #pastS = tangled[j].copy()
            rightS = pastS * (knotR - evalPoints)
            tangled[j] = (leftS + rightS)
            leftS = pastS * (evalPoints - knotL)
        tangled[i+1] = leftS


    # tangled[0] = np.ones(numx) / (knots[left+1] - knots[left])
    # for i in range(deg - 1):  # loop up the degrees
    #     leftS = np.zeros(numx)
    #     j = 0
    #
    #     knotL = knots[left - i - 1]  # knots left
    #     for j in range(i + 1):  # loop over the tangled calculated splines
    #         knotR = knots[left + j + 1]  # knots right
    #         # Cox-De Boor formula
    #         pastS = tangled[j].copy()
    #         rightS = pastS * (knotR - evalPoints)
    #         tangled[j] = (leftS + rightS) / (knotR - knotL)
    #
    #         knotL = knots[left + (j - i)]  # knots left
    #         leftS = pastS * (evalPoints - knotL)
    #     j = j+1
    #     knotR = knots[left + j + 1]  # knots right
    #     tangled[j] = leftS / (knotR - knotL)
    #
    # for i in range(deg):
    #     tangled[i] = tangled[i] * deg



    """
    Unbundle the calculated values
    """
    # O(numx*(deg+numk)
    designMatrix = np.zeros((numx, nums))

    # O(numx*deg)
    rowStart = 0
    for s in range(nums):  # loop up the splines
        colStart = min(deg - 1, s)
        row = rowStart
        knotLast = knots[s]

        for col in range(colStart, -1, -1):  # loop down columns of tangled, include 0
            while row < numx and knots[left[row]] == knotLast:  # increase the row until the interval changed
                designMatrix[row, s] = tangled[col, row]  # transfer value to splineMat
                row += 1
            if row < numx:
                knotLast = knots[left[row]]  # knotLast tracks the interval changes
            if s >= (deg - 1) and col == colStart:  # on the last column, on the first iteration
                rowStart = row  # lower the rowStart

    return designMatrix


# Evaluates N-splines
#   basis splines constrained to have the 2nd derivative = 0 at the first and last knot.
#
# Inputs:
#   evalPoints:     list of points to evaluate the splines at   (sorted ascending)  # TODO: handle unsorted
#   deg:            degree of the splines                       (polynomial degree + 1)
#   knots:          list of knots                               (sorted ascending without duplicates)
#
#   bSplineMat:     (default None) Pre-evaluated B-Spline matrix. Overwrites evalPoints
#   positive:       (default False) Constrains the N-splines to be positive (does not change other properties)
#
# Output:
#   designMatrix:   first dim - evalPoints
#                   second dim - spline index  (number of splines = deg + len(knots) - 4)
#
# Algorithm:
#   evaluate B-splines
#   calculate the 2nd derivatives at the first and last knot
#       (uses the derivative formula for B-splines, simplified heavily)  TODO: proof
#   use the first and last B-spline to normalize the second derivatives to be 0
#
# Time Complexity:
#   not pre-evaluated:
#       O(len(evalPoints) * (deg^2 + len(knots)))     from calling bSplineMatrix
#   pre-evaluated:
#       O(len(evalPoints))
#
# Space Complexity:
#   O(len(evalPoints) * (deg + len(knots)))
def nSplineMatrix(evalPoints, order, knots, bSplineMat=None, positive=False):
    """
    calculate B-splines
    """
    # O(len(evalPoints) * (deg^2 + len(knots)))
    # if provided use the pre-evaluated bSpline matrix
    if bSplineMat is None:
        bSplineMat = bSplineMatrix(evalPoints, order, knots)

    # add padding to the knots
    knots = np.concatenate((np.ones(order - 1) * knots[0], knots, np.ones(order - 1) * knots[-1]))

    """
    calculate the second derivatives
    """
    # O(1)
    # calculate the second derivatives at the first knot for the first 3 splines
    degFac = (order - 1) * (order - 2)
    width1 = knots[order] - knots[0]
    width2 = knots[order + 1] - knots[1]
    s0 = degFac / width1 ** 2
    s1 = -degFac / width1 * (1 / width1 + 1 / width2)
    s2 = degFac / width1 / width2

    # second derivative at the last knot for the last 3 splines
    width1 = knots[-1 - order] - knots[-1]
    width2 = knots[-2 - order] - knots[-2]
    s_3 = degFac / width1 / width2
    s_2 = -degFac / width1 * (1 / width1 + 1 / width2)
    s_1 = degFac / width1 ** 2

    """
    normalize
    B-splines -> N-splines
    """
    # O(len(evalPoints))
    # cut off the first and last B-splines
    nSplineMat = bSplineMat[:, 1:-1]

    # using the first B-spline normalize the second derivatives to be 0
    nSplineMat[:, 0] -= s1 / s0 * bSplineMat[:, 0]
    nSplineMat[:, 1] -= s2 / s0 * bSplineMat[:, 0]
    if positive:  # optionally make the make all N-splines positive
        nSplineMat[:, 1] -= nSplineMat[0, 1] / nSplineMat[0, 0] * nSplineMat[:, 0]

    # using the last B-spline normalize the second derivatives to be 0
    nSplineMat[:, -2] -= s_3 / s_1 * bSplineMat[:, -1]
    nSplineMat[:, -1] -= s_2 / s_1 * bSplineMat[:, -1]
    if positive:  # optionally make the make all N-splines positive
        nSplineMat[:, -2] -= nSplineMat[-1, -2] / nSplineMat[-1, -1] * nSplineMat[:, -1]

    return nSplineMat


"""
test code
"""
# import matplotlib.pyplot as plt
#
# evalPoints = np.linspace(0,1,2000)
# degree = 4
# knots = np.linspace(0,1,5)
#
# B_Mat = bSplineMatrix(evalPoints,degree,knots)
# N_Mat = nSplineMatrix(evalPoints,degree,knots)
#
# plt.figure()
# for s in B_Mat.T:
#     plt.plot(evalPoints,s)
# plt.title("B-Splines")
#
# plt.figure()
# for s in N_Mat.T:
#     plt.plot(evalPoints,s)
# plt.title("N-Splines")
#
# plt.show()



"""
Analytic derivative formula
"""

#
# evalPoints = np.linspace(0,1,2000)
# order = 4
# knots = np.linspace(0,1,5)
#
# B3 = bSplineMatrix(evalPoints,order-1,knots)
# B4 = bSplineMatrix(evalPoints,order,knots)
#
#
# numDer = B4.copy()
# for j in range(0,B4.shape[1]):
#     for i in range(0,B4.shape[0]-1):
#         numDer[i,j] = (B4[i+1,j]-B4[i,j]) / (evalPoints[i+1] - evalPoints[i])
# anDer = B4.copy()
#
# degree = order-1
# knotsBuffered = np.concatenate((np.ones(order-1)*knots[0],knots,np.ones(order-1)*knots[-1]))
#
# for j in range(0,B4.shape[1]):
#     for i in range(0,B4.shape[0]):
#         widthLeft = knotsBuffered[j + degree] - knotsBuffered[j]
#         widthRight = knotsBuffered[j + 1 + degree] - knotsBuffered[j + 1]
#
#         if widthLeft == 0:
#             anDer[i,j] = - order * B3[i,j]/widthRight
#         elif widthRight == 0:
#             anDer[i,j] = order * B3[i,j-1]/widthLeft
#         else:
#             anDer[i, j] = order * (B3[i, j - 1] / widthLeft - B3[i,j] / widthRight)
#
#
# plt.figure()
# for s in B4.T:
#     plt.plot(evalPoints,s)
# plt.title("B-Splines")
#
#
# plt.figure()
# for s in numDer.T:
#     plt.plot(evalPoints,s)
# plt.title("Derivatives numerically calculated")
#
# plt.figure()
# for s in anDer.T:
#     plt.plot(evalPoints,s)
# plt.title("Derivatives analytically calculated")
#
#
# plt.show()
