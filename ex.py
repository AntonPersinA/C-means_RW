import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Генерация данных
np.random.seed(42)

# Здоровые пациенты
group1 = np.random.normal([5.5, 120, 22], [0.5, 5, 2], size=(100, 3))

# Риск гипертонии
group2 = np.random.normal([5.8, 140, 25], [0.6, 7, 3], size=(100, 3))

# Риск диабета
group3 = np.random.normal([8.5, 130, 30], [0.8, 6, 4], size=(100, 3))

# Объединяем данные
data = np.vstack((group1, group2, group3))

# Параметры синтетического набора
features = ['Глюкоза (ммоль/л)', 'Давление (мм рт. ст.)', 'ИМТ (кг/м²)']


def find_optimal_clusters(data, max_clusters=10):
    fpcs = []
    for n_clusters in range(2, max_clusters + 1):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data.T, n_clusters, 2.5, error=0.005, maxiter=1000, init=None
        )
        fpcs.append(fpc)

    plt.plot(range(2, max_clusters + 1), fpcs, marker='o')
    plt.title(" Elbow method C - means ")
    plt.xlabel(" Number of clusters ")
    plt.ylabel(" Fuzzy Partition Coefficient ( FPC ) ")
    plt.grid()
    plt.tight_layout()
    plt.show()
    return fpcs

# Применение метода локтя
fpcs = find_optimal_clusters(data)
c = 4
# Применение C-means
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data.T, c, 2.5, error=0.001, maxiter=1000, init=None
)

# Метки кластеров (максимальная вероятность)
labels = np.argmax(u, axis=0)

# Визуализация
for i in range(c):
    plt.scatter(data[labels == i, 2], data[labels == i, 1], label=f'Кластер {i+1}')

plt.scatter(cntr[:, 2], cntr[:, 1], c='red', marker='x', s=200, label='Центры кластеров')
plt.title("Результаты C-means кластеризации")
plt.xlabel(features[2])
plt.ylabel(features[1])
plt.legend()
plt.tight_layout()
plt.show()



import pandas as pd

pd.set_option('display.max_columns', None)

# Найти произвольного пациента из каждого кластера
examples = []
for i in range(c):
    # Найти индекс любого объекта из текущего кластера
    cluster_indices = np.where(labels == i)[0]
    random_idx = np.random.choice(cluster_indices)
    examples.append((random_idx, *data[random_idx]))

# Создать DataFrame для таблицы
columns = ['Индекс пациента', 'Глюкоза (ммоль/л)', 'Давление (мм рт. ст.)', 'ИМТ (кг/м²)']
examples_df = pd.DataFrame(examples, columns=columns)

# Добавление вероятностей принадлежности к каждому кластеру
probabilities = []
for example in examples:
    idx = example[0]  # Индекс пациента
    probs = u[:, idx]  # Вероятности принадлежности к каждому кластеру
    probabilities.append([idx, *probs])

# Создать DataFrame для вероятностей
columns_probabilities = ['Индекс пациента'] + [f'Кластер {i+1} (%)' for i in range(c)]
probabilities_df = pd.DataFrame(probabilities, columns=columns_probabilities)

# Объединение с основными данными
final_df = examples_df.merge(probabilities_df, on='Индекс пациента')

# Визуализация таблицы с помощью matplotlib
fig, ax = plt.subplots(figsize=(11, 1.5))
ax.axis('tight')
ax.axis('off')
table_data = final_df.round(2).values.tolist()
table_columns = list(final_df.columns)
table = ax.table(cellText=table_data, colLabels=table_columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(table_columns))))
plt.title("Примеры пациентов из каждого кластера", fontsize=14)
plt.tight_layout()
plt.show()

# Визуализация данных примеров пациентов
for i in range(c):
    plt.scatter(data[labels == i, 2], data[labels == i, 1], label=f'Кластер {i+1}')

plt.scatter(cntr[:, 2], cntr[:, 1], c='red', marker='x', s=200, label='Центры кластеров')
for i, row in examples_df.iterrows():
    plt.annotate(
        f'Пациент {int(row["Индекс пациента"])}',
        (row['ИМТ (кг/м²)'], row['Давление (мм рт. ст.)']),
        textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='blue'
    )
    plt.scatter(row['ИМТ (кг/м²)'], row['Давление (мм рт. ст.)'], c='blue', edgecolor='black', s=100, label=f'Пример пациента {i+1}')

plt.title("Результаты C-means кластеризации с примерами пациентов")
plt.xlabel(features[2])
plt.ylabel(features[1])
plt.legend()
plt.tight_layout()
plt.show()