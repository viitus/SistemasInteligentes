import pandas as pd
# print ("Pandas version:", pd.__version__)
dados = pd.read_csv('dadosAulav1.csv', sep=';', encoding='latin1')

print("-------------Dados originais:-----------------")
print(dados.head())


# -----------REMOVENDO LINHAS---------------------------------------------
#       Vendo quantos linhas podem ser removidas
print("\n\nRemoção de linhas")
print("Quantidade de linhas duplicadas:")
print(dados.duplicated().sum())
#       Removendo linhas duplicadas
dados = dados.drop_duplicates()
print("Quantidade de linhas duplicadas após remoção:")
print(dados.duplicated().sum())
#       Verificando se há linhas com valores faltantes
faltantesPorLinha = dados.isnull().sum(axis=1)
print("Contagem de valores faltantes por linha > 1:")
print(faltantesPorLinha[faltantesPorLinha > 1])
#       Descobri que somente uma linha tem mais de 2 valores faltantes
#       Removendo a linha com mais de 2 valores faltantes
dados = dados[faltantesPorLinha <= 2]
faltantesPorLinha = dados.isnull().sum(axis=1)
print("Contagem de valores faltantes por linha > 1 após correção:")
print(faltantesPorLinha[faltantesPorLinha > 1])


#----------Corrigindo a coluna 'Doença'--------------------------------
print("\n\nCorrigindo a coluna 'Doença'")
print("Valores únicos na coluna 'Doença':")
print(dados['Doença'].unique())
#       Em relação aos dados, a informação sobre a "Doença" que estiver vazia pode ser considerada como “Saudável”. 
dados['Doença'] = dados['Doença'].fillna('Saudável')
print("Valores únicos na coluna 'Doença' após correção:")
print(dados['Doença'].unique())
doencaHotOne = pd.get_dummies(dados['Doença'], prefix='Doença').astype(int)
dados = pd.concat([dados, doencaHotOne], axis=1)
print("Colunas após a criação de variáveis dummy para 'Doença':")
print(list(dados.columns))
# print(dados[['Doença', 'Doença_Saudável', 'Doença_A', 'Doença_B', 'Doença_C']].head())


#----------Corrigindo a coluna 'Pressão'-------------------------------
print("\n\nCorrigindo a coluna 'Pressão'")
print("Valores únicos na coluna 'Pressão':")
print(dados['Pressão'].unique())
modaPressao = dados['Pressão'].mode()[0]
dados['Pressão'] = dados['Pressão'].fillna(modaPressao)
print("Valores únicos na coluna 'Pressão' após correção:")
print(dados['Pressão'].unique())
dados['Pressão'] = dados['Pressão'].map({
    'Baixa': 0,
    'Boa': 1,
    'Alta': 2
})
print("Valores únicos na coluna 'Pressão' após mapeamento:")
print(dados['Pressão'].unique())


#----------Corrigindo a coluna 'Mancha'-------------------------------------------
print("\n\nCorrigindo a coluna 'Mancha'")
print("Valores únicos na coluna 'Mancha':")
print(dados['Mancha'].unique())
dados['Mancha'] = dados['Mancha'].str.upper()
print("Valores únicos na coluna 'Mancha' após conversão para maiúsculas:")
print(dados['Mancha'].unique())
dados['Mancha'] = dados['Mancha'].replace({
    'SIM': 'Sim',
    'S': 'Sim',
    'NÃO': 'Não',
    'NAO': 'Não',
    'N': 'Não'
})
print("Valores únicos na coluna 'Mancha' após substituição:")
print(dados['Mancha'].unique())
manchaHotOne = pd.get_dummies(dados['Mancha'], prefix='Mancha').astype(int)
dados = pd.concat([dados, manchaHotOne], axis=1)
print("Colunas após a criação de variáveis dummy para 'Mancha':")
print(list(dados.columns))
# print(dados[['Mancha', 'Mancha_Sim', 'Mancha_Não']].head())


#----------Corrigindo a coluna 'Idade' e 'Data de Nascimento'----------------------
#   Contando quantas linhas estão faltando a 'Idade' e 'Data de Nascimento'
print("\n\nCorrigindo a coluna 'Idade' e 'Data de Nascimento'")
print("Valores maximos e mínimos da coluna 'Idade':")
print(dados['Idade'].max())
print(dados['Idade'].min())
# Valor máximo é 230 o que é um valor inválido para idade, pois a pessoa estaria morta.
# Substituindo o valor máximo inválido por NaN
dados['Idade'] = dados['Idade'].replace(230, pd.NA)
# Verificando novamente os valores máximos e mínimos
print("Valor máximos coluna 'Idade' após remoção:")
print(dados['Idade'].max())
print("Quantidade de valores faltantes na coluna 'Idade':")
print(dados['Idade'].isnull().sum())
# Preenchendo 'Idade' onde está faltando, usando 'Data de Nascimento'
# Criando a máscara para linhas onde 'Idade' é nula e 'Data de Nascimento' não é nula
mascara = (dados['Idade'].isnull()) & (dados['Data de Nascimento'].notnull())
# Convertendo a 'Data de Nascimento' para datetime e extraindo o ano
anos_nascimento = pd.to_datetime(dados.loc[mascara, 'Data de Nascimento'], errors='coerce').dt.year
# Calculando e preenchendo a 'Idade'
dados.loc[mascara, 'Idade'] = 2025 - anos_nascimento
print("Quantidade de valores faltantes na coluna 'Idade' após preenchimento com 'Data de Nascimento':")
print(dados['Idade'].isnull().sum())
# Preencher as idades restantes com a média
mediaIdade = dados['Idade'].mean()
dados['Idade'] = pd.to_numeric(dados['Idade'], errors='coerce')  # Garantindo que a coluna seja numérica
dados['Idade'] = dados['Idade'].fillna(mediaIdade)
print("Quantidade de valores faltantes na coluna 'Idade' após preenchimento com a média:")
print(dados['Idade'].isnull().sum())


#----------Corrigindo a coluna 'Febre'------------------------------------------------------------------------
# Encontrando o maior e menor valor da coluna 'Febre'
print("\n\nCorrigindo a coluna 'Febre'")
print("Valores máximos e mínimos da coluna 'Febre':")
print(dados['Febre'].max())
# O maior valor é 46.0, o que é um valor inválido para febre, pois a pessoa estaria morta.
print(dados['Febre'].min())
# O menor valor é 26.0, o que é um valor inválido para febre, ainda mais que a pessoa é considerada saudável. 
# Portanto, vamos substituir esses valores por NaN para que possamos tratá-lo posteriormente.
dados['Febre'] = dados['Febre'].replace(26.0, pd.NA)
dados['Febre'] = dados['Febre'].replace(46.0, pd.NA)
# Verificando novamente os valores máximos e mínimos
print("Valores máximos e mínimos da coluna 'Febre' após substituição:")
print(dados['Febre'].max())
print(dados['Febre'].min())
# Agora os valores máximos e mínimos são válidos, pois o máximo é 40.0 e o mínimo é 35.0.

#   Corrigindo a coluna 'Febre' onde está faltando e estão saudáveis 
print("Quantidade de valores faltantes na coluna 'Febre' antes da correção:")
print(dados['Febre'].isnull().sum())
media_febre_saudavel = dados.loc[dados['Doença'] == 'Saudável', 'Febre'].mean()
mascara_febre_saudavel = (dados['Doença'] == 'Saudável') & (dados['Febre'].isnull())
dados.loc[mascara_febre_saudavel, 'Febre'] = media_febre_saudavel
print("Quantidade de valores faltantes na coluna 'Febre' após a correção dos saudáveis:")
print(dados['Febre'].isnull().sum())
#   Corrigindo a coluna 'Febre' onde está faltando e não estão saudáveis 
media_febre_nao_saudavel = dados.loc[dados['Doença'] != 'Saudável', 'Febre'].mean()
mascara_febre_nao_saudavel = (dados['Doença'] != 'Saudável') & (dados['Febre'].isnull())
dados.loc[mascara_febre_nao_saudavel, 'Febre'] = media_febre_nao_saudavel
print("Quantidade de valores faltantes na coluna 'Febre' após a correção final:")
print(dados['Febre'].isnull().sum())


#-----------Removendo colunas desnecessárias-----------------------------
print("\n\nRemovendo colunas desnecessárias")
print("Colunas antes da remoção:")
print(list(dados.columns))
dados = dados.drop(columns=['Nome', 'Data de Nascimento', 'Doença', 'Mancha', 'Doença'])
print("Colunas após a remoção:")
print(list(dados.columns))

# Verificação de valores faltantes
print("\n\nVerificando valores faltantes após todas as correções:")
print(dados.isnull().sum())
# Nenhum valor faltante encontrado após as correções


#-----------Normalizando os dados-----------------------------------
colunas_para_normalizar = ['Idade', 'Febre', 'Pressão']

for coluna in colunas_para_normalizar:
    min_val = dados[coluna].min()
    max_val = dados[coluna].max()
    dados[coluna] = (dados[coluna] - min_val) / (max_val - min_val)

# Verificando o resultado
#print(dados[colunas_para_normalizar].head())
print("\n\n----------------------Dados processados:--------------------")
print(dados.head())
# Salvando o DataFrame processado em um novo arquivo CSV
dados.to_csv('dadosAula1Processados.csv', index=False, sep=';')

