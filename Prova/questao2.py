# Questão 02 – Considere os dados apresentados abaixo. 
# Escolha um dos algoritmos estudados em sala de aula para resolver esse problema.


from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
# Gerar dados
X, y = make_moons(n_samples=500, noise=0.06, random_state=23)
# Visualizar
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title("Two Moons Dataset")
plt.show()



import math

# -------------------Etapa 1 inicialização dos clusters-------------------

# funcao de calculo da distancia entre dois pontos
def distancia_pontos(ponto1, ponto2):
    return math.sqrt((ponto1[0] - ponto2[0]) ** 2 + (ponto1[1] - ponto2[1]) ** 2)

# encontrar a menor distancia entre dois clusters
def distancia_clusters(cluster1, cluster2):
    menor_distancia = float('inf')
    for ponto1 in cluster1:
        for ponto2 in cluster2:
            dist = distancia_pontos(ponto1, ponto2)
            if dist < menor_distancia:
                menor_distancia = dist
    return menor_distancia

# -------------------Etapa 2 agrupamento usando o Single-Link -------------------

def agrupamento_single_link(dados, k):
    # Inicializar clusters com cada ponto como um cluster separado
    clusters = [[ponto] for ponto in dados]
    
    # Enquanto o número de clusters for maior que k, unir os mais próximos
    while len(clusters) > k:
        print(f"Número de clusters atual: {len(clusters)}")
        menor_distancia = float('inf')
        clusters_proximos = (0, 0)
        
        # encontrar os dois clusters mais próximos
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = distancia_clusters(clusters[i], clusters[j])
                if dist < menor_distancia:
                    #encontrou a menor distancia
                    menor_distancia = dist
                    clusters_proximos = (i, j)
        
        # encorporar j em i e remover j
        i, j = clusters_proximos 
        clusters[i].extend(clusters[j])
        del clusters[j]
    return clusters


# -------------------Etapa 3 visualização dos clusters-------------------
k = 2
clusters = agrupamento_single_link(X, k)

for i, cluster in enumerate(clusters):
    plt.scatter(*zip(*cluster), label=f'Cluster {i+1}')
plt.title("Clusters Resultantes usando Single-Link")
plt.legend()
plt.show()