import numpy as np
import matplotlib.pyplot as plt
from skfuzzy.cluster import cmeans

# Функция генерации данных в виде вложенных колец
def generate_elliptical_rings(n_points, n_rings, noise=0.1):
    X = []
    y = []
    for i in range(1, n_rings + 1):
        theta = np.linspace(0, 2 * np.pi, n_points)
        a = i  # Большая полуось
        b = i / 2  # Малая полуось
        x_ring = a * np.cos(theta) + noise * np.random.randn(n_points)
        y_ring = b * np.sin(theta) + noise * np.random.randn(n_points)
        X.append(np.column_stack((x_ring, y_ring)))
        y.append(np.full(n_points, i))
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

# Генерация данных
n_points = 300  # Точек на кольце
n_rings = 3
noise = 0.05
X, y_true = generate_elliptical_rings(n_points, n_rings, noise)


# Визуализация
plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10', s=10)
plt.title("Generated Elliptical Rings Clusters")
plt.xlabel("X1")
plt.ylabel("X2")
plt.axis("equal")
plt.tight_layout()
plt.show()



# Транспонируем X для совместимости с c-means
X_transposed = X.T

# Параметры fuzzy c-means
n_clusters = n_rings  # Количество кластеров соответствует количеству колец
delta = 0.005  # Допустимая ошибка для завершения алгоритма
max_iter = 1000
m = 2.0

# Запуск fuzzy c-means
cntr, u, _, _, _, _, _ = cmeans(
    X_transposed,
    c=n_clusters,
    m=m,
    error=delta,
    maxiter=max_iter,
    init=None  # Автоматическая инициализация центров
)

# Присвоение каждой точки наиболее вероятного кластера
cluster_labels = np.argmax(u, axis=0)

# Визуализация результата
plt.figure(figsize=(8, 8))
for i in range(n_clusters):
    plt.scatter(
        X[cluster_labels == i, 0],
        X[cluster_labels == i, 1],
        label=f'Cluster {i+1}'
    )

plt.scatter(cntr[:, 0], cntr[:, 1], c='red', marker='x', s=200, label='Centers')
plt.title("Fuzzy C-Means Clustering")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()
