# Plotar Sobrevivência por tamanho da família
plt.figure(figsize=(10, 6))
dados.groupby('FamilySize')['Survived'].mean().plot(kind='bar', color='purple', alpha=0.7)
plt.title('Taxa de Sobrevivência por Tamanho da Família')
plt.xlabel('Tamanho da Família')
plt.ylabel('Taxa de Sobrevivência')
plt.axhline(dados['Survived'].mean(), color='red', linestyle='dashed', linewidth=1, label='Média de Sobrevivência')
plt.legend()
plt.show()