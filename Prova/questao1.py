# QUESTAO 01 – Desenvolva um algoritmo FP-Growth para resolver o problema 
# Market_Basket_Optimisation (disponível juntamente com a prova). 
# Utilize um suporte mínimo de 300 e confiança mínima de 0,3. 
# Apresente os dados analisados, destacando:
#• Os itens que mais aparecem individualmente nas transações;
#• Os conjuntos de itens mais frequentes que ocorrem juntos.

import csv

SUPORTEMINIMO = 300
CONFIANCAMINIMA = 0.3

dados = 'Prova/Market_Basket_Optimisation.csv'

# -------------------Carregar os dados do arquivo CSV-------------------

def carregar_dados(dados):
    with open(dados, 'r') as d:
        leitor = csv.reader(d)
        transacoes = [linha for linha in leitor]
    return transacoes

transacoes = carregar_dados(dados)

# print para verificar se foi carregado corretamente
#print("\nPrimeiras linhas:----------------------------------------")
#for t in transacoes[:5]:
#    print(t)


# -------------------Etapa 1 contar a frequencia de cada item individualmente-------------------

def contar_itens(transacoes):
    contagem = {}
    for transacao in transacoes:
        for item in transacao:
            if item in contagem:
                contagem[item] += 1
            else:
                contagem[item] = 1
    return contagem

#print("\nContagem de itens individuais:----------------------------------------")
#for item in contar_itens(transacoes).items():
#    print(f"{item[0]}: {item[1]}")


# -------------------Etapa 2 contar a frequencia consicederando o suporte mínimo-------------------

contagem_itens = contar_itens(transacoes)

def filtrar_itens(contagem, SUPORTEMINIMO):
    itens_frequentes = {item: count for item, count in contagem.items() if count >= SUPORTEMINIMO}
    return itens_frequentes 

itens_frequentes = filtrar_itens(contagem_itens, SUPORTEMINIMO)

#print("\nItens com frequencia > 300 :----------------------------------------")
#for item in itens_frequentes.items():
#    print(f"{item[0]}: {item[1]}")

# -------------------Etapa 3 reordenar as transações com base nos itens frequentes-------------------

def reordenar_transacoes(transacoes, itens_frequentes):
    transacoes_reordenadas = []
    for transacao in transacoes:
        transacao_filtrada = [item for item in transacao if item in itens_frequentes]
        if transacao_filtrada:
            transacoes_reordenadas.append(transacao_filtrada)
    return transacoes_reordenadas

transacoes_reordenadas = reordenar_transacoes(transacoes, itens_frequentes)

#print("\nTransações reordenadas:----------------------------------------")
#for t in transacoes_reordenadas[:5]:
#    print(t)


#-------------------- Etapa 4 criar estrutura de dados em uma árvore FP-------------------------

class NoFP:
    def __init__(self, item, contador, pai=None):
        self.item = item
        self.contador = contador
        self.pai = pai
        self.filhos = {}
        self.proximo = None

    def incrementar_contador(self):
        self.contador += 1

class ArvoreFP:
    def __init__(self):
        self.raiz = NoFP(None, 0)
        self.itens_frequentes = {}

    def adicionar_transacao(self, transacao):
        no_atual = self.raiz
        for item in transacao:       
            if item in no_atual.filhos:
                no_atual.filhos[item].incrementar_contador()
            else:
                novo_no = NoFP(item, 1, no_atual)
                no_atual.filhos[item] = novo_no
                if item in self.itens_frequentes:
                    atual = self.itens_frequentes[item]
                    while atual.proximo:
                        atual = atual.proximo
                    atual.proximo = novo_no
                else:
                    self.itens_frequentes[item] = novo_no
            no_atual = no_atual.filhos[item]

    def construir_arvore(self, transacoes):
        for transacao in transacoes:
            self.adicionar_transacao(transacao)

arvore_fp = ArvoreFP()
arvore_fp.construir_arvore(transacoes_reordenadas)

#ex: dado a transação [ground beef,pepper,spaghetti,cookies]
#a arvore sera construida raiz-> ground beef(1) -> pepper(1) -> spaghetti(1) -> cookies(1)
#onde o numero em parenteses representa o contador de ocorrências do item
#entao se houver outra transação com esses itens cada um sera incrementado em 1
#se houver uma trnsacao com itens diferentes sera criada uma nova ramificação


# -----------Etapa 5 extrair os conjuntos de itens frequentes----------------------

# extrai os caminhos condicionais a partir dos itens frequentes
def caminhos_condicionais(item, itens_frequentes):
    caminhos = []
    no = itens_frequentes.get(item)
    while no is not None:
        caminho = []
        pai = no.pai
        while pai is not None and pai.item is not None:
            caminho.append(pai.item)
            pai = pai.pai
        if caminho:
            caminhos.append((list(reversed(caminho)), no.contador))
        no = no.proximo
    return caminhos


# ------------------- Etapa 6 construir a árvore condicional -------------------

# controir uma árvore condicional a partir dos caminhos condicionais e suporte mínimo
def construir_arvore_condicional(caminhos, SUPORTEMINIMO):
    contagem = {}
    for caminho, cont in caminhos:
        for item in caminho:
            contagem[item] = contagem.get(item, 0) + cont
    itens_validos = {item for item, count in contagem.items() if count >= SUPORTEMINIMO}
    transacoes_filtradas = []
    for caminho, cont in caminhos:
        caminho_filtrado = [item for item in caminho if item in itens_validos]
        caminho_filtrado.sort(key=lambda x: contagem[x], reverse=True)
        if caminho_filtrado:
            transacoes_filtradas.extend([caminho_filtrado] * cont)
    arvore = ArvoreFP()
    arvore.construir_arvore(transacoes_filtradas)
    return arvore


# ------------------- Etapa 7 minerar os padrões frequentes -------------------

# extrai os padrões da arvore com frequencia maior que o suporte mínimo
def minerar_fp(arvore, prefixo, SUPORTEMINIMO):
    padroes = []
    itens = sorted(arvore.itens_frequentes.items(), key=lambda x: x[1].contador)
    for item, no in itens:
        novo_padrao = prefixo + [item]
        suporte = 0
        temp = no
        while temp is not None:
            suporte += temp.contador
            temp = temp.proximo
        if suporte >= SUPORTEMINIMO:
            padroes.append((novo_padrao, suporte))
            caminhos = caminhos_condicionais(item, arvore.itens_frequentes)
            arvore_condicional = construir_arvore_condicional(caminhos, SUPORTEMINIMO)
            padroes += minerar_fp(arvore_condicional, novo_padrao, SUPORTEMINIMO)
    return padroes

padroes_frequentes = minerar_fp(arvore_fp, [], SUPORTEMINIMO)

def calcular_confianca(padroes, CONFIANCAMINIMA):
    confiancas = []
    for padrao, suporte in padroes:
        if len(padrao) > 1:  
            for item in padrao:
                suporte_item = contar_itens(transacoes).get(item, 0)
                if suporte_item > 0:
                    confianca = suporte / suporte_item
                    if confianca >= CONFIANCAMINIMA:
                        confiancas.append((padrao, confianca))
    return confiancas


# ------------------- Resultados Finais -----------------------------------------------------

print("\n------------------- Itens mais frequentes (individulmente) -------------------")
for item, count in sorted(contagem_itens.items(), key=lambda x: x[1], reverse=True):
    if count >= SUPORTEMINIMO:
        print(f"{item} = {count}")


print("\n------------------- Conjuntos de itens com maior frequência -------------------")
for padrao, suporte in sorted(padroes_frequentes, key=lambda x: x[1], reverse=True):
    if len(padrao) > 1:
        print(f"{padrao} = {suporte}")


confiancas = calcular_confianca(padroes_frequentes, CONFIANCAMINIMA)
print("\n------------------- Conjuntos com confiança > 0.3 -------------------")
for padrao, confianca in confiancas:
    print(f"{padrao} = {confianca:.2f}")
print("\n")