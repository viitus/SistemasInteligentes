import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.svm import SVC

from sklearn.datasets import load_iris

# Carregar o dataset Iris
iris = load_iris()

X = iris.data[:, :2]
y = iris.target

# Definindo nomes das features e classes
feature_names = iris.feature_names[:2]
target_names = iris.target_names

# Definindo colormaps
cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#0000FF', '#FF0000', '#00FF00'])

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Dicionário para armazenar resultados interessantes
resultados = []

# ----------------------------- Treinar modelo Árvore de Decisão -------------------------
modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# Gerar relatório para a árvore de decisão
with open("relatorio_arvore.txt", "w", encoding="utf-8") as f:
    f.write("Relatório - Árvore de Decisão (2 features)\n")
    f.write("Features usadas: " + ", ".join(feature_names) + "\n\n")
    f.write("Matriz de Confusão:\n")
    f.write(str(confusion_matrix(y_test, y_pred)) + "\n\n")
    f.write("Relatório:\n")
    f.write(classification_report(y_test, y_pred, target_names=target_names))

# Gerar gráfico de decisão
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Prever a classe para cada ponto na grade
Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotar a fronteira de decisão
plt.figure(figsize=(6, 4))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', label='Treino')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, edgecolor='k', marker='*', label='Teste')
plt.title('Árvore de Decisão - Iris Dataset (2 features)')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.legend()
plt.show()

resultados.append({
    'modelo': 'Árvore de Decisão',
    'Acurácia': modelo.score(X_test, y_test),
})


# ----------------------------- Treinar modelo Naive Bayes -------------------------
modelo = GaussianNB()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# Gerar relatório para o Naive Bayes
with open('relatorio_naive_bayes.txt', 'w', encoding='utf-8') as f:
    f.write("Relatório - Naive Bayes (2 features)\n")
    f.write("Features usadas: " + ", ".join(feature_names) + "\n\n")
    f.write("Matriz de Confusão:\n")
    f.write(str(confusion_matrix(y_test, y_pred)) + "\n\n")
    f.write("Relatório de Classificação:\n")
    f.write(classification_report(y_test, y_pred, target_names=target_names))

# Gerar gráfico de decisão para o Naive Bayes
h = 0.02  
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

# Prever a classe para cada ponto na grade
Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotar a fronteira de decisão
plt.figure(figsize=(6, 4))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', label='Treino')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, edgecolor='k', marker='*', label='Teste')
plt.title('Naive Bayes - Iris Dataset (2 features)')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.legend()
plt.show()

resultados.append({
    'modelo': 'Naive Bayes',
    'Acurácia': modelo.score(X_test, y_test),
})

# ----------------------------- Treinar modelo KNN -------------------------

for k in [1, 3, 5]:

    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Gerar relatório para o KNN
    with open(f'relatorio_knn_k{k}.txt', 'w', encoding='utf-8') as f:
        f.write(f"Relatório - KNN (k={k})\n")
        f.write("Features usadas: " + ", ".join(feature_names) + "\n\n")
        f.write("Matriz de Confusão:\n")
        f.write(str(confusion_matrix(y_test, y_pred)) + "\n\n")
        f.write("Relatório de Classificação:\n")
        f.write(classification_report(y_test, y_pred, target_names=target_names))

    # Gerar gráfico de decisão para o KNN
    Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plotar a fronteira de decisão
    plt.figure(figsize=(6, 4))
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    plt.title(f'KNN - Iris Dataset (k={k})')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', label='Treino')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, edgecolor='k', marker='*', label='Teste')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend()
    plt.show()

    resultados.append({
        'modelo': f'KNN (k={k})',
        'Acurácia': modelo.score(X_test, y_test),
    })


# ------------------------------ Treinar modelo SVM -------------------------


modelo = SVC(kernel='linear')
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# Gerar relatório para o SVM
with open('relatorio_svm.txt', 'w', encoding='utf-8') as f:
    f.write("Relatório - SVM (2 features)\n")
    f.write("Features usadas: " + ", ".join(feature_names) + "\n\n")
    f.write("Matriz de Confusão:\n")
    f.write(str(confusion_matrix(y_test, y_pred)) + "\n\n")
    f.write("Relatório de Classificação:\n")
    f.write(classification_report(y_test, y_pred, target_names=target_names))

# Gerar gráfico de decisão para o SVM
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Prever a classe para cada ponto na grade
Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotar a fronteira de decisão
plt.figure(figsize=(6, 4))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', label='Treino')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, edgecolor='k', marker='*', label='Teste')
plt.title('SVM - Iris Dataset (2 features)')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.legend()
plt.show()

resultados.append({
    'modelo': 'SVM',
    'Acurácia': modelo.score(X_test, y_test),
})


# ---------------------------- Comparar os modelos treinados ----------------------------

print("Resultados dos Modelos Treinados:\n")
for resultado in resultados:
    print(f"{resultado['modelo']}: Acurácia = {resultado['Acurácia']:.2f}") 
print ("\nO modelo com melhor acurácia foi: ", max(resultados, key=lambda x: x['Acurácia'])['modelo'])