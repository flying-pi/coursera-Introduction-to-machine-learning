import numpy as np

X = np.random.normal(loc=1, scale=10, size=(7, 5))
print("random matrix :: \n")
print(X)
print("\n\n")

std = np.std(X, axis=0)
m = np.mean(X, axis=0)
X_norm = ((X - m) / std)
print("normalized random matrix :: ")
print(X_norm)
print("\n\n")

Z = np.array([[4, 5, 0],
              [1, 9, 3],
              [5, 1, 1],
              [3, 3, 3],
              [9, 9, 9],
              [4, 7, 1]])
Z_more10 = np.empty((0, 3), int)
print("row of matrix, which sum > 10 :: ")
r = np.sum(Z, axis=1)
for i in range(r.size):
    if r[i] > 10:
        Z_more10 = np.vstack((Z_more10, [Z[i, :]]))
print(Z_more10)
print("\n\n")

print("Concatenation of matrix: ")
A = np.eye(3)
B = np.eye(3)

print("A: ")
print(A)
print("B: ")
print(B)
print("result1 :")
print(np.vstack((A, B)))
print("result2 :")
print(np.hstack((A, B)))
