import pandas as pd

# Carregar o dataset
dados = pd.read_csv('titanic.csv')

# Verificar valores faltantes
print(dados.isnull().sum())

# Preencher 'Age' com mediana
dados['Age'] = dados['Age'].fillna(dados['Age'].median())

# Criar coluna: tem cabine ou não
dados['HasCabin'] = dados['Cabin'].notnull().astype(int)

# Remover a coluna 'Cabin'
dados = dados.drop(columns=['Cabin'])

# Criar a nova coluna 'FamilySize'
dados['FamilySize'] = dados['SibSp'] + dados['Parch']

# Remover colunas desnecessárias
dados = dados.drop(columns=['Name', 'PassengerId', 'SibSp', 'Parch'])


# Resultados estatísticos
print("Estatísticas descritivas:")
# Mediana de idades
print(f"Mediana de idades: {dados['Age'].median()}")
# Média de sobrevivência
print(f"Média de sobrevivência: {dados['Survived'].mean()}")
# Numero de sobreviventes
print(f"Número de sobreviventes: {dados['Survived'].sum()}")
# Número de homens que sobreviveram
num_homens_sobreviveram = dados[(dados['Sex'] == 'male') & (dados['Survived'] == 1)].shape[0]
print(f"Número de homens que sobreviveram: {num_homens_sobreviveram}")
# Número de mulheres que sobreviveram
num_mulheres_sobreviveram = dados[(dados['Sex'] == 'female') & (dados['Survived'] == 1)].shape[0]
print(f"Número de mulheres que sobreviveram: {num_mulheres_sobreviveram}")



# Plotar gráficos para análise visual
import matplotlib.pyplot as plt

# Plotar a distribuição de idades
plt.figure(figsize=(10, 6))
plt.hist(dados['Age'], bins=30, color='blue', alpha=0.7)
plt.title('Distribuição de Idades')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.axvline(dados['Age'].median(), color='red', linestyle='dashed', linewidth=1, label='Mediana')
plt.legend()
plt.show()

# Plotar a distribuição do tamanho da família
plt.figure(figsize=(10, 6))
plt.hist(dados['FamilySize'], bins=range(0, dados['FamilySize'].max() + 2), color='green', alpha=0.7)
plt.title('Distribuição do Tamanho da Família')
plt.xlabel('Tamanho da Família')
plt.ylabel('Frequência')
plt.show()

# Plotar Sobrevivência por tamanho da família
plt.figure(figsize=(10, 6))
dados.groupby('FamilySize')['Survived'].mean().plot(kind='bar', color='purple', alpha=0.7)
plt.title('Taxa de Sobrevivência por Tamanho da Família')
plt.xlabel('Tamanho da Família')
plt.ylabel('Taxa de Sobrevivência')
plt.axhline(dados['Survived'].mean(), color='red', linestyle='dashed', linewidth=1, label='Média de Sobrevivência')
plt.legend()
plt.show()

# Plotar Sobrevivência por Classe
plt.figure(figsize=(10, 6))
dados.groupby('Pclass')['Survived'].mean().plot(kind='bar', color='skyblue', alpha=0.8)
plt.title('Taxa de Sobrevivência por Classe')
plt.xlabel('Classe')
plt.ylabel('Taxa de Sobrevivência')
plt.axhline(dados['Survived'].mean(), color='red', linestyle='dashed', linewidth=1, label='Média Geral')
plt.legend()
plt.show()

# Salvar o dataset processado
dados.to_csv('titanicProcessados.csv', index=False)