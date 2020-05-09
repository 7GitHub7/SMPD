import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy.spatial import distance


# Zbiór uczący (wiersze - próbki, kolumny - cechy - można zrobić odwrotnie, ale trzeba też zmienić niżej)
X = np.array([
    [1, 1, 1],
    [1, 2, 1],
    [1, 3, 2],
    [1, 4, 3],
    [2, 4, 2],
    [2, 3, 3],
    [2, 2, 1],
    [1, 4, 2]
])
Y = np.array([
    [3, 5, 2],
    [10, 3, 1],
    [4, 7, 3],
    [5, 4, 2],
    [8, 4, 3],
    [9, 5, 2],
    [6, 4, 3],
    [7, 7, 1]
])


# Zbiór testowy
test = np.array([
    [2, 2, 2],
    [7, 5, 2]
])

# Obliczenie macierzy kowiariancji (gotowa funkcja z numpy)
X_cov = np.cov(X.T)
Y_cov = np.cov(Y.T)
print(f'Macierz kowariancji dla zbioru X:\n{pd.DataFrame({"cecha 1": X_cov[:, 0], "cecha 2": X_cov[:, 1], "cecha 3": X_cov[:, 2]}, index=["cecha 1", "cecha 2", "cecha 3"])}\n')
print(f'Macierz kowariancji dla zbioru Y:\n{pd.DataFrame({"cecha 1": Y_cov[:, 0], "cecha 2": Y_cov[:, 1], "cecha 3": Y_cov[:, 2]}, index=["cecha 1", "cecha 2", "cecha 3"])}\n')



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Dane')
ax.set_xlabel('cecha 1')
ax.set_ylabel('cecha 2')
ax.set_zlabel('cecha 3')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='red', label='X')
for x in X:
    ax.text(x[0], x[1], x[2], 'X')
ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color='blue', label='Y')
for y in Y:
    ax.text(y[0], y[1], y[2], 'Y')
ax.scatter(test[:, 0], test[:, 1], test[:, 2], color='green', marker='^', s=100, label='test')

plt.legend(loc=2)

for point in test:
    dist = []
    for gr_cov, gr, gr_name in zip([X_cov, Y_cov], [X, Y], ['X', 'Y']):
        gr_mean = np.mean(gr, axis=0)
        gr_cov_inv = np.linalg.inv(gr_cov)
        dist += [distance.mahalanobis(point, gr_mean, gr_cov_inv)]
        print(f'Odległość próbki {point} do grupy {gr_name} wynosi {dist[-1]:.4}.')
    ax.text(point[0], point[1], point[2], ['X', 'Y'][np.argmin(dist)], fontweight='bold')
plt.show()    