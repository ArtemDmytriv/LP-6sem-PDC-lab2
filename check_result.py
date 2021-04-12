import sympy as sp
import numpy as np

def checkResult():
    def readMatrix(f, n):
        mat = np.loadtxt(f, max_rows=n)
        return sp.Matrix(mat)

    f = open("matrices.txt", 'r')
    n = int(f.readline().split()[0])

    A = readMatrix(f, n)
    bi = sp.Matrix([[8/i] for i in range(1, n+1)])

    A1 = readMatrix(f, n)
    b1 = readMatrix(f, n)
    c1 = readMatrix(f, n)

    A2 = readMatrix(f, n)
    B2 = readMatrix(f, n)
    C2 = sp.Matrix( [[ 1/(i + j + 2) for j in range(1, n + 1)] for i in range(1, n + 1)])

    y1 = A*bi
    y2 = A1*(2*b1 + 3*c1)
    Y3 = A2*(B2 - C2)

    res = (y1*(y1.T)*Y3*y2*(y2.T) + Y3**2 + y1*(y2.T))*(y1*(y2.T)*Y3*y2 + y1)
    return res