#Questão 03 – Considere os dados apresentados abaixo. 
# Escolha um dos algoritmos estudados em sala de aula para resolver esse problema.

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
# Gerar dados artificiais com 4 grupos
X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=1.5, random_state=23)
# Visualizar os clusters
plt.scatter(X[:, 0], X[:, 1])
plt.title("Dados")
plt.show()


import numpy as np


# -------------------Etapa 1 inicialização dos centroides-------------------
 
def inicializar_centroides(dados, k):
    # Seleciona k pontos aleatórios como centros iniciais
    indices = np.random.choice(len(dados), k, replace=False)
    return dados[indices]

# -------------------Etapa 2 atribuição de clusters-------------------

def atribuir_clusters(dados, centroides):
    # Atribui cada ponto ao centroide mais próximo
    distancias = np.linalg.norm(dados[:, np.newaxis] - centroides, axis=2)
    return np.argmin(distancias, axis=1)

# -------------------Etapa 3 atualização dos centroides-------------------

def atualizar_centroides(dados, clusters, k):
    # Calcula novos centroides como a média dos pontos em cada cluster
    centroides = []
    for i in range(k):
        pontos_cluster = dados[clusters == i]
        centroide = pontos_cluster.mean(axis=0)
        centroides.append(centroide)
    return np.array(centroides)

# -------------------Etapa 4 K-medias (iteração)-------------------

def kmeans(dados, k, max_iter=100): 
    centroides = inicializar_centroides(dados, k)
    for _ in range(max_iter):
        # Atribui clusters e atualiza centroides
        clusters = atribuir_clusters(dados, centroides)
        novos_centroides = atualizar_centroides(dados, clusters, k)
        # Verifica se os centroides não mudaram
        if np.all(centroides == novos_centroides):
            break
        centroides = novos_centroides
    return clusters, centroides

# -------------------Etapa 5 Método do Cotovelo-------------------

def calcular_inercia(dados, clusters, centroides):
    # Calcula a inércia como a soma das distâncias quadradas dos pontos aos seus centroides
    inercia = 0
    for i in range(len(centroides)):
        pontos_cluster = dados[clusters == i]
        diferencas = pontos_cluster - centroides[i]
        distancias_ao_quadrado = np.sum(diferencas ** 2)
        inercia += distancias_ao_quadrado
    return inercia


def metodo_do_cotovelo(dados, max_k=10):
    inercia = []
    for k in range(1, max_k + 1):
        clusters, centroides = kmeans(dados, k)
        inercia.append(calcular_inercia(dados, clusters, centroides))
    return inercia

inercia = metodo_do_cotovelo(X, max_k=10)

plt.plot(range(1, 11), inercia, marker='o')
plt.title("Método do Cotovelo")
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Inércia")
plt.xticks(range(1, 11))
plt.grid()
plt.show()


for k in range(2, 7):
    clusters, centroides = kmeans(X, k)
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', marker='o')
    plt.scatter(centroides[:, 0], centroides[:, 1], c='red', marker='x', s=200, label='Centroides')
    plt.title(f"K-means com (k={k})")
    plt.legend()
    plt.show()