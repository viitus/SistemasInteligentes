import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# Carregar dados
dados = pd.read_csv('dados_azul_laranja.csv')
X = dados[['x1', 'x2']]
y = dados['Classe']

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Definir colormap
cmap_light = ListedColormap(['#AAAAFF', '#FFAAAA'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

# Grid para o gráfico
h = 0.02  # passo da malha
x_min, x_max = X['x1'].min() - 1, X['x1'].max() + 1
y_min, y_max = X['x2'].min() - 1, X['x2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))



#-------------------- Definir o modelo KNN e treinar com K = 1, 2 e 3

# Testar para K = 1, 2, 3
for k in [1, 2, 3]:
    # Treinar modelo
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Relatórios
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    with open(f'relatorio_k{k}.txt', 'w', encoding='utf-8') as f:
        f.write(f"Relatório para K = {k}\n")
        f.write("Matriz de Confusão:\n")
        f.write(str(cm) + "\n\n")
        f.write("Relatório de Classificação:\n")
        f.write(cr)

    # Fronteira de decisão
    Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
    mapa_classes = {'Azul': 0, 'Laranja': 1}
    Z_num = np.array([mapa_classes[c] for c in Z])
    Z_num = Z_num.reshape(xx.shape)

    plt.figure(figsize=(6, 4))
    plt.contourf(xx, yy, Z_num, cmap=cmap_light, alpha=0.8)

    # Plotar pontos de treino
    plt.scatter(X_train['x1'], X_train['x2'], c=y_train.map({'Azul': 1, 'Laranja': 0}), 
                cmap=cmap_bold, edgecolor='k', s=60, label='Treino')
    # Plotar pontos de teste
    plt.scatter(X_test['x1'], X_test['x2'], c=y_test.map({'Azul': 1, 'Laranja': 0}),
                cmap=cmap_bold, edgecolor='k', s=60, marker='x', label='Teste')

    plt.title(f"Fronteira de Decisão (K = {k})")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.tight_layout()
    plt.show()

#-------------------- Definir o modelo Arvore de Decisão e treinar


# Treinar modelo de árvore de decisão
arvore = DecisionTreeClassifier()
arvore.fit(X_train, y_train)

# Prever com o modelo de árvore de decisão
y_pred_arvore = arvore.predict(X_test)

# Relatórios para árvore de decisão
cm_arvore = confusion_matrix(y_test, y_pred_arvore)
cr_arvore = classification_report(y_test, y_pred_arvore)

with open('relatorio_arvore.txt', 'w', encoding='utf-8') as f:
    f.write("Relatório para Árvore de Decisão\n")
    f.write("Matriz de Confusão:\n")
    f.write(str(cm_arvore) + "\n\n")
    f.write("Relatório de Classificação:\n")
    f.write(cr_arvore)

# Fronteira de decisão para árvore de decisão
Z_arvore = arvore.predict(np.c_[xx.ravel(), yy.ravel()])
Z_arvore_num = np.array([mapa_classes[c] for c in Z_arvore])
Z_arvore_num = Z_arvore_num.reshape(xx.shape)

plt.figure(figsize=(6, 4))
plt.contourf(xx, yy, Z_arvore_num, cmap=cmap_light, alpha=0.8)

# Plotar pontos de treino
plt.scatter(X_train['x1'], X_train['x2'], c=y_train.map({'Azul': 1, 'Laranja': 0}), 
            cmap=cmap_bold, edgecolor='k', s=60, label='Treino')

# Plotar pontos de teste
plt.scatter(X_test['x1'], X_test['x2'], c=y_test.map({'Azul': 1, 'Laranja': 0}),
            cmap=cmap_bold, edgecolor='k', s=60, marker='x', label='Teste')

plt.title("Fronteira de Decisão (Árvore de Decisão)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.tight_layout()
plt.show()
