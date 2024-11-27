import numpy as np
import matplotlib.pyplot as plt
from skfuzzy.cluster import cmeans

# Функция генерации двух прямых
def generate_lines(n_points, line_params, noise=0.1):
    """
    Генерирует точки вдоль двух прямых.
    :param n_points: количество точек на каждой линии
    :param line_params: список параметров прямых [((x1, y1), (x2, y2)), ...]
    :param noise: уровень шума
    :return: X, y (координаты точек и метки кластеров)
    """
    X = []
    y = []
    for i, ((x1, y1), (x2, y2)) in enumerate(line_params):
        t = np.linspace(0, 1, n_points)
        x_line = x1 + t * (x2 - x1) + noise * np.random.randn(n_points)
        y_line = y1 + t * (y2 - y1) + noise * np.random.randn(n_points)
        X.append(np.column_stack((x_line, y_line)))
        y.append(np.full(n_points, i))
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

# Генерация данных для двух прямых
n_points = 300  # Количество точек на каждой линии
line_params = [((0, 0), (0, 5)), ((2, 5), (2, 0))]  # Параметры прямых (конечные точки)
noise = 0.1
X, y_true = generate_lines(n_points, line_params, noise)

# Визуализация исходных данных
plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10', s=10)
plt.title("Generated Line Clusters")
plt.xlabel("X1")
plt.ylabel("X2")
plt.axis("equal")
plt.tight_layout()
plt.show()

# Транспонируем X для совместимости с c-means
X_transposed = X.T

# Параметры fuzzy c-means
n_clusters = len(line_params)  # Количество кластеров соответствует числу прямых
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

# Визуализация результата кластеризации
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
