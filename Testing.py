from copy import copy

import numpy as np
import matplotlib.pyplot as plt

import Spline as S
import SplineFunctions as G


def integral(y,l):
    i = 0
    for j in range(1,len(y)):
        i += 0.5 *(y[j-1]+y[j])*l
    return i

def integrate(y,l):
    result = [0]*len(y)
    for i in range(1,len(y)):
        result[i] = result[i-1] + 0.5* (y[i-1]+y[i])*l
    return result

def derivative(y,l):
    d = np.zeros(len(y))
    d[0] = (y[1]-y[0])/l
    for i in range(1,len(y)-1):
        d[i] = (y[i+1]-y[i-1])/(2*l)
    d[-1] = (y[-1]-y[-2])/l
    return d

def GramMatrix(y,l):
    n = y.shape[0]
    G = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            G[i,j] = integral(np.multiply(y[i,:],y[j,:]),l)
    return G

def GramSchmidt(y,l):
    TOL = 0.001
    h = y.shape[0]
    for i in range(h):
        for j in range(i):
            y[i,:] = y[i,:] - integral(np.multiply(y[i,:],y[j,:]),l)*y[j,:]
        norm = np.sqrt(integral(np.multiply(y[i,:],y[i,:]),l))
        if norm > 0.001:
            y[i,:] /= norm
            # y[i, :] *= np.sign(integral(y[i, :], l))
        else:
            y[i, :] *= 0
    return y

def GramSchmidtAdd(y,l):
    TOL = 0.001
    h = y.shape[0]
    for i in range(h):
        for j in range(i):
            y[i,:] = y[i,:] - integral(y[i,:]+y[j,:],l)*y[j,:]
        norm = integral(y[i,:],l)
        y[i,:] = (y[i,:]-norm) *np.sign(norm)
    return y




#
# evalPoints = np.linspace(0,4,40000)
# knots = np.linspace(0,4,5)
# degree = 3
#
# y = S.bSplineMatrix(evalPoints,degree,knots)
#
#
# plt.figure()
# for s in y.T:
#     plt.plot(evalPoints,s)
#
# print(GramMatrix(y.T,1/10000))
#
# plt.figure()
# for s in y.T:
#     plt.plot(evalPoints,integrate(s,1/400))
#
#
#
#
# plt.show()




'''
GenerateSplineFunctions Test:
'''


# back end functions

# parameters needed
# degree = 4
# k = 6
# knots = np.linspace(0,k-1,k)
# resolution = 50
# x = np.linspace(0,k-1,(k-1)*resolution+1)  # evaluation points tied together with the parameters
#
# y1 = G.M(degree,knots,resolution)  # M splines
# plt.figure()
# for f in y1:
#     plt.plot(x,f)
# plt.title("GenerateSplineFunction Backend: M")
#
# y2 = G.I(degree,knots,resolution)  # I splines
# plt.figure()
# for f in y2:
#     plt.plot(x,f)
# plt.title("GenerateSplineFunction Backend: I")
#
# for s in y1:
#     integral = [0] * len(s)
#     for i in range(len(s)-1):
#         integral[i+1] = integral[i] + s[i]*(x[i+1]-x[i])
#     plt.plot(x,integral)




#plt.show()

# front end function

# # parameters needed
# degree = 4
# k = 6
# knots = np.linspace(0,k-1,k)
#
# basisFun = G.getSplineBasisFun(degree,knots,True)  # returns basis function for I-splines
#
# # evaluation points (not tied to the parameters)
# resolution = 500
# x = np.linspace(0,k-1,resolution)
#
# n = degree-1 + k-1  # number of functions
#
# plt.figure()
# for f in range(n):
#     y = [basisFun(f,i) for i in x]
#     plt.plot(x,y)
# plt.title("GenerateSplineFunction I-basis Function")




'''
CoefficientPathImplementation Test:
'''


# # parameters needed
# order = 4
# k = 20
# knots = list(np.linspace(0,1,k))
#
# coefficientsB = C.B(order,knots)  # coefficient matrix
# # print the coefficients with better format
# print("[")
# for row in coefficientsB:
#     print("\t[",end='')
#     for i in range(len(row)-1):
#         print("%.3f, " % row[i], end='')
#     print("%.3f" % row[-1], end='')
#     print("]")
# print("]")
# # TODO: documentation
#
# coefficientsN = C.N(order,knots,positive=False)  # coefficient matrix
# # print the coefficients with better format
# print("[")
# for row in coefficientsN:
#     print("\t[",end='')
#     for i in range(len(row)-1):
#         print("%.3f, " % row[i], end='')
#     print("%.3f" % row[-1], end='')
#     print("]")
# print("]")
#
# coefficientsN = C.N(order,knots,positive=True)  # coefficient matrix
# # print the coefficients with better format
# print("[")
# for row in coefficientsN:
#     print("\t[",end='')
#     for i in range(len(row)-1):
#         print("%.3f, " % row[i], end='')
#     print("%.3f" % row[-1], end='')
#     print("]")
# print("]")
# # TODO: documentation
#
# # evaluation points (not tied to the parameters)
# resolution = 500
# x = list(np.linspace(0,1,resolution))
#
# # bmat = S.bSplineMatrix(x,order,knots)
#
#
# plt.figure()
# for spline in coefficientsN:
#     y = C.spEval(knots,spline,x)
#     plt.plot(x,y)
# plt.title("CoefficientPathImplementation M-basis")
#
#
# # for spline in bmat.T:
# #     plt.plot(x,spline)
# # plt.title("Spline M-basis")
#
# knots = np.pad(knots, order-1, 'edge')
# deg = order
# degFac = (deg - 1) * (deg - 2)
# width1 = knots[deg] - knots[0]
# width2 = knots[deg + 1] - knots[1]
# s0 = degFac / width1 ** 2
# s1 = -degFac / width1 * (1 / width1 + 1 / width2)
# s2 = degFac / width1 / width2
#
# print(s0,s1,s2)
#
# # second derivative at the last knot for the last 3 splines
# width1 = knots[-1 - deg] - knots[-1]
# width2 = knots[-2 - deg] - knots[-2]
# s_3 = degFac / width1 / width2
# s_2 = -degFac / width1 * (1 / width1 + 1 / width2)
# s_1 = degFac / width1 ** 2
#
# print(s_3,s_2,s_1)
#
# #
# # coefficientsI = C.I(order,knots)  # coefficient matrix
# # # print the coefficients with better format
# # print("[")
# # for row in coefficientsI:
# #     print("\t[",end='')
# #     for i in range(len(row)-1):
# #         print("%.3f, " % row[i], end='')
# #     print("%.3f" % row[-1], end='')
# #     print("]")
# # print("]")
# # # TODO: documentation
# #
# # plt.figure()
# # for spline in coefficientsI:
# #     y = C.spEval(knots,spline,x)
# #     plt.plot(x,y)
# # plt.title("CoefficientPathImplementation I-basis")
# #
# plt.show()



# degree = 4
# k = 10
# n = k + degree - 2
# end = k-1
# knots = list(np.linspace(0,end,k))
# UB = np.identity(n)
#
# res = 50000
# x = list(np.linspace(0,end,res))
# y = np.zeros((n,res))
#
# for i in range(UB.shape[0]):
#     y[i,:] = C.spEval(knots,list(UB[i,:]),x)
#
# # for i in range(UB.shape[0]-1):
# #     y[i,:] = y[i,:]-y[i+1,:]
#
# print(y)
#
# g = GramMatrix(y,end/res)
# for row in g:
#     print("\t[",end='')
#     for i in range(len(row)-1):
#         print("%.3f, " % row[i], end='')
#     print("%.3f" % row[-1], end='')
#     print("]")
# print("]")


#
# order = 5
#
# k = 20
# end = 1
# degree = order - 1
# n = k + degree - 1
# knots = list(np.linspace(0,end,k))
# der = 2
#
# print(C.derivativesReshaped(order, knots, der))
# d = C.derivatives(order, knots, der)
# print(d)
#
# for row in d:
#     print("\t[",end='')
#     for i in range(len(row)-1):
#         print("%.4f, " % row[i], end='')
#     print("%.4f" % row[-1], end='')
#     print("]")
# print("]")
#
# g = C.gram(order-der, knots)
# print(g)
#
# for row in g:
#     print("\t[",end='')
#     for i in range(len(row)-1):
#         print("%.4f, " % row[i], end='')
#     print("%.4f" % row[-1], end='')
#     print("]")
# print("]")
#
#
# p = C.BpenaltyMatrixTangled(order,knots)
#
# for row in p:
#     print("\t[",end='')
#     for i in range(len(row)-1):
#         print("%.4f, " % row[i], end='')
#     print("%.4f" % row[-1], end='')
#     print("]")
# print("]")
#
# print(C.BpenaltyMatrixTangled(order,knots,sparse=True))
#
#
# p = C.NpenaltyMatrixTangled(order,knots)
#
# for row in p:
#     print("\t[",end='')
#     for i in range(len(row)-1):
#         print("%.4f, " % row[i], end='')
#     print("%.4f" % row[-1], end='')
#     print("]")
# print("]")


# res = 5000
# x = list(np.linspace(0,end,res))
# y = S.bSplineMatrix(x, order, knots)
#
# plt.figure()
# for sp in y.T:
#     plt.plot(x,sp)
#
# plt.figure()
# for sp in y.T:
#     plt.plot(x,derivative(derivative(sp,end/res),end/res))
#
#
# res = 5000
# x = list(np.linspace(0,end,res))
# y = S.bSplineMatrix(x, order-2, knots)
#
#
# plt.figure()
# for sp in (y@d).T:
#     plt.plot(x,sp)
#
# plt.show()







# res = 50000
# x = list(np.linspace(0,end,res))
# y = np.zeros((n,res))
#
# UB = C.B(order,knots)
#
# for i in range(n):
#     y[i,:] = C.spEval(knots,list(UB[i,:]),x)
#
# # for i in range(UB.shape[0]-1):
# #     y[i,:] = y[i,:]-y[i+1,:]
#
# g = GramMatrix(y,end/res)
# for row in g:
#     print("\t[",end='')
#     for i in range(len(row)-1):
#         print("%.6f, " % row[i], end='')
#     print("%.6f" % row[-1], end='')
#     print("]")
# print("]")
#
# g2 = C.gram(order,knots)
#
# for row in g2:
#     print("\t[",end='')
#     for i in range(len(row)-1):
#         print("%.6f, " % row[i], end='')
#     print("%.6f" % row[-1], end='')
#     print("]")
# print("]")
#
# for row in g2-g:
#     print("\t[",end='')
#     for i in range(len(row)-1):
#         print("%.6f, " % row[i], end='')
#     print("%.6f" % row[-1], end='')
#     print("]")
# print("]")




# p = C.BpenaltyMatrix(order,knots)
# for row in p:
#     print("\t[",end='')
#     for i in range(len(row)-1):
#         print("%.2f, " % row[i], end='')
#     print("%.2f" % row[-1], end='')
#     print("]")

# UB = np.eye(n)
# for r in UB:
#     y = C.spEval(knots,list(r),list(x))
#     plt.plot(x,y)
#
# plt.title("Partial Truncated Basis, Order 4, 6 Knots")
#
# plt.show()


order = 4

k = 6
end = 1
degree = order - 1
n = k + degree - 1
knots = list([-2,-1.9,0,1,2,3])
der = 2
res = 200
x = np.linspace(-5,0,res)

p = C.BpenaltyMatrix(order,knots,2)
print(p.shape)
for row in p:
    print("\t[",end='')
    for i in range(len(row)-1):
        print("%.2f, " % row[i], end='')
    print("%.2f" % row[-1], end='')
    print("]")
print("")

d2 = G.penaltyMatrix(order,knots,der=2,sparse=False,BSpline=True)
print(d2.shape)
for row in d2:
    print("\t[",end='')
    for i in range(len(row)-1):
        print("%.2f, " % row[i], end='')
    print("%.2f" % row[-1], end='')
    print("]")

#
# b = S.bSplineMatrix(x,order,knots)
# plt.figure()
# for y in b.T:
#     plt.plot(x,y)
#
#
# b = G.BCoeffs(order,knots)
# for row in b:
#     print("\t[",end='')
#     for i in range(len(row)-1):
#         print("%.2f, " % row[i], end='')
#     print("%.2f" % row[-1], end='')
#     print("]")
#
# plt.figure()
# for sp in b:
#     y = C.spEval(knots,list(sp),x)
#     plt.plot(x,y)
# plt.show()

# bMat = S.bSplineMatrix(x,order,knots)
# for y in bMat.T:
#     plt.plot(x,y)
#
# plt.title("Penalized Fitting")
#
# c = np.array([4,2,5,2,6,4,8,6,5,8,7,3,5,2,5,2,5,4])
# y = c@bMat.T
#
#
#
# plt.plot(x,y,color='black',linestyle='dashed')
#
#
# x = np.linspace(0,end,40)
# bMat = S.bSplineMatrix(x,order,knots)
#
# y2 = c@bMat.T + 2*np.random.random(40) - 1
# plt.scatter(x,y2)
#
#
# x = np.linspace(0,end,res)
# bMat = S.bSplineMatrix(x,order,knots)
# c = np.array([3.5,3,3,4,4.5,5.5,6.5,7,7,7,6.5,5,3.25,3,3.25,3.5,4,4.25])
# y3 = c@bMat.T
#
# plt.plot(x,y3,color = 'red')





plt.show()
