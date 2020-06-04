#!bin/python3

from math import copysign, hypot
import numpy as np
import pprint
from matplotlib import pyplot as plt

X = [[8, 9, 8, 8], 
    [6, 5, 6, 6], 
    [6, 7, 6, 9], 
    [6, 8, 7, 10]]#, 
    # [6, 8, 5, 10]]

"""

Reference: https://en.wikipedia.org/wiki/QR_decomposition

This code is a part of the algorithm that is used to find
the non-negative matrix factorization.

All the functions given below is used to find the QR 
decompopsition of a given matrix which will be used 
for the factorization of matrix later.

QR decomposition:
    A = QR
    where R is an upper triangular matrix and Q is
    an orthogonal matrix, i.e., one satisfying
    Transpose(Q) *  Q = I 

QR decomposition is often used to solve the linear
least squares problem. Also used for Hierarchical Least
squares(HALS).
"""

def gramSchmidt(X):
    """
    Returns Q and R matrix for given X using graph_schimidt process.
    But numerically unstable due to orthogonalization.
    """
    # rows, columns = np.shape(X)

    # Q = np.empty([rows, rows])
    # count = 0

    # for a in X.T:
    #     u = np.copy(a)
    #     for i in range(0, count):
    #         projection = np.dot(np.dot(Q[:, i].T, a), Q[:, i])
    #         u -= projection

    #     e = u / np.linalg.norm(u)
    #     Q[:, count] = e
    #     count += 1

    # R = np.dot(Q.T, X)

    # return (Q, R)
    Q, R = np.linalg.qr(X)
    return (Q, R)


def houseHolder(X):
    """
    Returns Q and R matrix for given X using house holder reflection process.
    Numericallly stable but bandwidth heavy(Requires more computation).
    """
    rows, columns = np.shape(X)

    Q = np.identity(rows)
    R = np.copy(X)

    for count in range(rows - 1):
        x = R[count:, count]

        e = np.zeros_like(x)
        e[0] = copysign(np.linalg.norm(x), -X[count, count])
        u = x + e
        v = u / np.linalg.norm(u)

        Q_count = np.identity(rows)
        Q_count[count:, count:] -= 2.0 * np.outer(v, v)

        R = np.dot(Q_count, R)
        Q = np.dot(Q, Q_count.T)

    return (Q, R)


def givensRotation(X):
    """
    Returns Q and R matrix for given X using house holder reflection process
    """
    nrows, ncolumns = np.shape(X)

    Q = np.identity(nrows)
    R = np.copy(X)

    rows, cols = np.tril_indices(nrows, -1, ncolumns)
    for row, col in zip(rows, cols):

        if R[row, col] != 0:
            (c, s) = givensEntries(R[col, col], R[row, col])

            G = np.identity(nrows)
            G[[col, row], [col, row]] = c
            G[row, col], G[col, row] = s, -s

            R = np.dot(G, R)
            Q = np.dot(Q, G.T)

    return (np.matrix(Q), np.matrix(R))

def givensEntries(a, b):
    r = hypot(a, b)
    return (a / r, -b / r)

# Return norm(X - Y)
def frobenius(X, Y):
    D = X - Y
    return np.linalg.norm(D, 'fro') 

# Select algo for QR decompositions
def QR(X, algo = 2):
    if algo == 2:
        return gramSchmidt(X)
    elif algo == 1:
        return houseHolder(X)
    return givensRotation(X)


# NMF model
class NMF:
    def __init__(self, X, _rank):
        # X is of shape m x n
        self.X = np.matrix(X)
        self.rank = _rank
        self.shape = self.X.shape

    def compute(self, p = 3, q = 2, iterations = 10, error = 100):
        # Slight oversampling
        l = self.rank + p
        
        # Generate sampling matrix of shape n x l
        O = np.random.rand(self.shape[1], l)
        
        # Form basis of shape m x l
        Y = self.X * O

        print(Y.shape)

        for i in range(q):
            Q, _ = QR(Y)
            Q, _ = QR(self.X.T * Q)
            Y = self.X * Q

        # Form orthogonal basis m x l
        # print('x', self.X.shape)
        Q, _ = QR(Y)
        # print('q', Q.shape)
        B = Q.T * self.X

        W, WT, H = np.random.rand(self.shape[0], self.rank), np.random.rand(l, self.rank), np.random.rand(self.rank, self.shape[1])
        W, WT, H = np.matrix(W), np.matrix(WT), np.matrix(H)

        # Higher the iterations higher the accuracy
        while iterations:
            iterations -= 1

            R = B.T * WT
            S = WT.T * WT

            # Update H row by row
            for j in range(self.rank):
                row = (R[:, j] - H.T * S[:, j]) / S[j, j]
                flatten = []
                for i in row:
                    flatten.append(i[0, 0])

                flatten = np.matrix(flatten)
                H[j, :] = H[j, :] + flatten

                # Element wise maximum to avoid small negative values
                for idx in range(len(H[j])):
                    H[j, idx] = max(0, H[j, idx])

            T = B * H.T
            V = H * H.T

            # Update W column wise
            for j in range(self.rank):
                WT[:, j] = WT[:, j] + (T[:, j] - WT * V[:, j]) / V[j, j]
                
                delta = (Q * W[:, j])
                for idx in range(len(W)):
                    W[idx, j] = max(0, delta[idx, 0])

                # Rotate to lower dimension
                WT[:, j] = Q.T * W[:, j]

            # Check the distance between original matrix and W * H
            if frobenius(self.X, W * H) <= error:
                break
                
        return (W, H)

nmf = NMF(X, 3)
W, H = nmf.compute(1)
print(W, H, sep='\n')

fig, axs = plt.subplots(3)
axs[0].imshow(X)
axs[0].axis("off")
axs[1].imshow(W)
axs[1].axis("off")
axs[2].imshow(H)
axs[2].axis("off")

plt.show()

img = plt.imread('image.jpeg')
# plt.axis("off")
# plt.imshow(img)
# plt.show()

# def rgb2gray(rgb):
#     return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# img = (rgb2gray(img))
# print(img)

# for i in range(1, 20):
#     print(i)
#     nmf = NMF(img, i)
#     for j in range(1, 10):
#         for k in range(1, 10):
#             nmf.compute(j, k)
#             try:
#                 nmf.compute(j, k)
#                 print("HIIIIIII")
#             except:
#                 print(j, k, 'not working')

#     # try:
#     print('try with 10 20')
#     nmf.compute(10, 20)
#     # except:
#     #     print('not working', i)