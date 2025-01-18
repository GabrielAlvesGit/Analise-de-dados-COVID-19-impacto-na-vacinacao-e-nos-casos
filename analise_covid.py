import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from statsmodels.stats.proportion import proportions_ztest

# 1. Carregar os dados
covid_data = pd.read_csv('owid-covid-data.csv')
vax_data = pd.read_csv('vaccinations.csv')

# 2. Análise de Dados de COVID-19 e Vacinação
# Vamos focar na comparação de mortes por milhão antes e depois da vacinação

# Filtrar dados para 2020 e 2021
covid_data['date'] = pd.to_datetime(covid_data['date'])
antes = covid_data[covid_data['date'].dt.year == 2020].groupby('location')['new_deaths_per_million'].mean()
depois = covid_data[covid_data['date'].dt.year == 2021].groupby('location')['new_deaths_per_million'].mean()

# Criar DataFrame pareado (apenas países que têm dados em ambos os períodos)
dados = pd.DataFrame({
    'pais': antes.index,
    'antes': antes.values,
    'depois': depois[antes.index].values
})

# Remover valores NaN
dados = dados.dropna()

# Estatísticas resumidas
print("\nEstatísticas resumidas para 'antes':\n", dados['antes'].describe())
print("\nEstatísticas resumidas para 'depois':\n", dados['depois'].describe())

# a. Boxplot dos dados antes e depois
plt.figure(figsize=(6, 6))
sns.boxplot(data=[dados['antes'], dados['depois']], width=0.3)
plt.xticks([0, 1], ['Antes', 'Depois'])
plt.title('Boxplot - Mortes por Milhão Antes vs Depois da Vacinação')
plt.ylabel('Mortes por Milhão')
plt.show()

# b. Histogramas
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(dados['antes'], bins=10, color='skyblue', edgecolor='black')
plt.title('Histograma - Antes da Vacinação')
plt.xlabel('Mortes por Milhão')
plt.ylabel('Frequência')

plt.subplot(1, 2, 2)
plt.hist(dados['depois'], bins=10, color='lightcoral', edgecolor='black')
plt.title('Histograma - Depois da Vacinação')
plt.xlabel('Mortes por Milhão')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()

# c. QQ plots - Verificam visualmente a normalidade
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
stats.probplot(dados['antes'], dist="norm", plot=ax[0])
ax[0].set_title('Gráfico QQ - Antes')
stats.probplot(dados['depois'], dist="norm", plot=ax[1])
ax[1].set_title('Gráfico QQ - Depois')
plt.show()

# Teste de normalidade Shapiro-Wilk
shapiro_antes = stats.shapiro(dados['antes'])
shapiro_depois = stats.shapiro(dados['depois'])
print("Teste Shapiro-Wilk - Antes: W =", shapiro_antes.statistic, ", p-valor =", shapiro_antes.pvalue)
print("Teste Shapiro-Wilk - Depois: W =", shapiro_depois.statistic, ", p-valor =", shapiro_depois.pvalue)

# d. Teste de igualdade de variâncias (Levene)
f_stat, f_pvalue = stats.levene(dados['antes'], dados['depois'])
print("\nTeste de Levene para variâncias: Estatística F =", f_stat, ", p-valor =", f_pvalue)

# e. Teste t pareado
t_stat, t_pvalue = stats.ttest_rel(dados['antes'], dados['depois'], alternative='two-sided')
print("\nTeste t pareado: Estatística t =", t_stat, ", p-valor =", t_pvalue)

# Conclusão
alpha = 0.05  # Nível de significância
if t_pvalue < alpha:
    print("Rejeitamos a hipótese nula: há uma diferença significativa entre 'antes' e 'depois'.")
else:
    print("Não rejeitamos a hipótese nula: não há diferença significativa entre 'antes' e 'depois'.")

# 3. Análise de Proporções
x = np.array([94, 113])  # Sucessos
n = np.array([125, 175])  # Total de tentativas
alpha = 0.05  # Nível de significância

# Teste de proporções unilateral
stat, p_value = proportions_ztest(count=x, nobs=n, alternative='larger')

# Resultados do teste
print(f"Estatística Z: {stat:.3f}")
print(f"p-valor: {p_value:.5f}")
print("Conclusão:", "Rejeita H0 - aceitamos a hipótese alternativa, ou seja, a proporção de sucesso da amostra X é maior que da Y.")

# Cálculo do intervalo de rejeição
p_comb = (x[0] + x[1]) / (n[0] + n[1])  # Proporção combinada
se = np.sqrt(p_comb * (1 - p_comb) * (1 / n[0] + 1 / n[1]))  # Erro padrão
lrr = -np.inf  # Limite inferior da região de rejeição
urr = norm.ppf(1 - alpha, loc=0, scale=se)  # Limite superior

# Gráfico do teste de hipótese (unilateral)
d_values = np.linspace(-0.1, 0.1, 1000)  # Ajuste o intervalo de valores
density = norm.pdf(d_values, loc=0, scale=se)
rejection_region = np.where(d_values > urr, density, 0)

plt.figure(figsize=(10, 6))
plt.plot(d_values, density, label="Densidade da distribuição")
plt.fill_between(d_values, rejection_region, color='red', alpha=0.3, label="Região de rejeição")
observed_difference = (x[0] / n[0]) - (x[1] / n[1])  # Diferença observada
plt.axvline(x=observed_difference, color='blue', linestyle='--', label="Diferença observada")
plt.axvline(x=0, color='black', linestyle='-', label="H0: Diferença = 0")
plt.title("Teste de Hipótese para Diferenças entre Proporções (Unilateral)")
plt.xlabel("Diferença de Proporções")
plt.ylabel("Densidade")
plt.legend()
plt.grid()
plt.show()