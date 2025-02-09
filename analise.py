import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.stats import chi2_contingency

# Carregar o dataframe
df = pd.read_csv("merged_filtered_data.csv")

def descritiva():
    # Coloca 0 nos valores faltantes
    df.fillna(0, inplace=True)
    
    # Estatística descritiva
    estatisticas_descritivas = df[["new_cases", "daily_vaccinations", "new_deaths", "people_vaccinated"]].describe()
    print("Estatísticas Descritivas:")
    print(estatisticas_descritivas)
    
    # Histogramas com contagem absoluta
    plt.figure(figsize=(12, 6))
    
    # Novos casos
    plt.subplot(1, 3, 1)
    sns.histplot(df["new_cases"], bins=30, kde=True, stat="count")  # Alterado para count
    plt.title("Histograma de Novos Casos")
    plt.xlabel("Novos Casos")
    plt.ylabel("Número de Dias")  # Alterado para clarificar a frequência
    
    # Vacinações diárias
    plt.subplot(1, 3, 2)
    sns.histplot(df["daily_vaccinations"], bins=30, kde=True, stat="count")  # Alterado para count
    plt.title("Histograma de Vacinações Diárias")
    plt.xlabel("Vacinações Diárias")
    plt.ylabel("Número de Dias")  # Alterado para clarificar a frequência
    
    # Novas mortes
    plt.subplot(1, 3, 3)
    sns.histplot(df["new_deaths"], bins=30, kde=True, stat="count")  # Alterado para count
    plt.title("Histograma de Novas Mortes")
    plt.xlabel("Novas Mortes")
    plt.ylabel("Número de Dias")  # Alterado para clarificar a frequência
    
    plt.tight_layout()
    plt.show()

def chi2():
    bins = 4
    df["cases_category"] = pd.cut(df["new_cases"], bins=bins, labels=False)
    df["deaths_category"] = pd.cut(df["new_deaths"], bins=bins, labels=False)
    
    # Tabela de contingência
    contingencia = pd.crosstab(df["cases_category"], [df["deaths_category"]])

    # Teste qui-quadrado
    qui2, p, _, _ = chi2_contingency(contingencia)
    
    # Visualizar com heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(contingencia, annot=True, cmap="Blues", fmt="d", cbar=False)
    plt.title("Distribuição de Novos Casos vs. Novas Mortes")
    plt.xlabel("Novas Mortes (Categorizadas)")
    plt.ylabel("Novos Casos (Categorizados)")
    plt.show()
    
    # Exibir resultados
    print("Tabela de Contingência:")
    print(contingencia)
    print(f"\nValor Qui-Quadrado: {qui2:.4f}")
    print(f"Valor p: {p:.4f}\n")
    
    # Conclusão
    if p < 0.05:
        print("Rejeitamos a hipótese nula: as variáveis não são independentes.")
    else:
        print("Não rejeitamos a hipótese nula: as variáveis são independentes.")

def probabilidade():
    # Probabilidade de registrar mais de 10 mil casos em um dia
    probabilidade_mais_10k_casos = stats.norm.sf(
        10000, df["new_cases"].mean(), df["new_cases"].std()
    )

    # Probabilidade de registrar mais de 100 mortes em um dia
    probabilidade_mais_100_mortes = stats.norm.sf(
        100, df["new_deaths"].mean(), df["new_deaths"].std()
    )

    # Probabilidade de vacinar mais de 100 mil pessoas em um dia
    probabilidade_mais_100k_vacinacao = stats.norm.sf(
        100000, df["daily_vaccinations"].mean(), df["daily_vaccinations"].std()
    )

    # Exibir resultados
    print(
        f"Probabilidade de registrar mais de 10 mil casos em um dia: {probabilidade_mais_10k_casos:.2f}\n"
    )
    print(
        f"Probabilidade de registrar mais de 100 mortes em um dia: {probabilidade_mais_100_mortes:.2f}\n"
    )
    print(
        f"Probabilidade de vacinar mais de 100 mil pessoas em um dia: {probabilidade_mais_100k_vacinacao:.2f}\n"
    )


def inferencia():
    # Dividir os dados antes, durante e depois do período de vacinação
    inicio_vacinacao = "2021-01-17"
    fim_vacinacao = "2023-03-22"
    df["date"] = pd.to_datetime(df["date"])
    antes = df[df["date"] < inicio_vacinacao]["new_cases"].dropna()
    durante = df[(df["date"] >= inicio_vacinacao) & (df["date"] <= fim_vacinacao)][
        "new_cases"
    ].dropna()
    depois = df[df["date"] > fim_vacinacao]["new_cases"].dropna()

    # Realizar o teste T entre antes e durante
    t_stat_antes_durante, p_value_antes_durante = stats.ttest_ind(antes, durante)

    # Realizar o teste T entre durante e depois
    t_stat_durante_depois, p_value_durante_depois = stats.ttest_ind(durante, depois)

    # Realizar ANOVA para comparar as médias dos três períodos
    f_stat, p_value_anova = stats.f_oneway(antes, durante, depois)

    # Exibir resultados
    print(
        f"Teste T (Antes vs Durante): estatística t = {t_stat_antes_durante:.2f}, valor p = {p_value_antes_durante:.2f}\n"
    )
    if p_value_antes_durante < 0.05:
        print(
            "Rejeitamos a hipótese nula: há uma diferença significativa entre as médias antes e durante o período de vacinação.\n"
        )
    else:
        print(
            "Não rejeitamos a hipótese nula: não há uma diferença significativa entre as médias antes e durante o período de vacinação.\n"
        )

    print(
        f"Teste T (Durante vs Depois): estatística t = {t_stat_durante_depois:.2f}, valor p = {p_value_durante_depois:.2f}\n"
    )
    if p_value_durante_depois < 0.05:
        print(
            "Rejeitamos a hipótese nula: há uma diferença significativa entre as médias durante e depois do período de vacinação.\n"
        )
    else:
        print(
            "Não rejeitamos a hipótese nula: não há uma diferença significativa entre as médias durante e depois do período de vacinação.\n"
        )

    print(f"ANOVA: estatística F = {f_stat:.2f}, valor p = {p_value_anova:.2f}\n")
    if p_value_anova < 0.05:
        print(
            "Rejeitamos a hipótese nula: há uma diferença significativa entre as médias dos três períodos.\n"
        )
    else:
        print(
            "Não rejeitamos a hipótese nula: não há uma diferença significativa entre as médias dos três períodos.\n"
        )

    # Visualizar as médias e intervalos de confiança
    data = {
        "Período": ["Antes", "Durante", "Depois"],
        "Média": [antes.mean(), durante.mean(), depois.mean()],
        "Desvio Padrão": [antes.std(), durante.std(), depois.std()],
    }
    df_inferencia = pd.DataFrame(data)
    df_inferencia["IC Inferior"] = df_inferencia["Média"] - 1.96 * (
        df_inferencia["Desvio Padrão"] / (len(antes) ** 0.5)
    )
    df_inferencia["IC Superior"] = df_inferencia["Média"] + 1.96 * (
        df_inferencia["Desvio Padrão"] / (len(antes) ** 0.5)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Período", y="Média", data=df_inferencia, errorbar=None)
    plt.errorbar(
        x=df_inferencia["Período"],
        y=df_inferencia["Média"],
        yerr=1.96 * (df_inferencia["Desvio Padrão"] / (len(antes) ** 0.5)),
        fmt="none",
        c="black",
        capsize=5,
    )
    plt.title(
        "Média de Novos Casos por Período com Intervalos de Confiança\n(Períodos: Antes: até 16/01/2021, Durante: 17/01/2021 a 22/03/2023, Depois: após 22/03/2023)"
    )
    plt.show()

def serie_temporal():
    # Converter a coluna de datas para o formato datetime
    df["date"] = pd.to_datetime(df["date"])

    # Filtrar dados até final 2023
    df_until_2023 = df[df["date"] <= "2023-05-14"]
    
    # Agrupar os dados por mês
    df_mensal = df_until_2023.resample("ME", on="date").sum()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=df_mensal.index, y=df_mensal["new_cases"] / 1000, label="Novos Casos (milhares)", color="blue")
    sns.lineplot(x=df_mensal.index, y=df_mensal["daily_vaccinations"] / 1000, label="Vacinações Diárias (milhares)", color="green")
    plt.title("Evolução de Novos Casos e Vacinação ao Longo do Tempo")
    plt.xlabel("Data")
    plt.ylabel("Quantidade (milhares)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

def correlacao():
    correlacao_pearson = df[["new_cases", "daily_vaccinations"]].corr(method="pearson")
    print("Correlação de Pearson entre novos casos e vacinação diária:")
    print(correlacao_pearson)

    # Mapa de calor para melhor visualização
    plt.figure(figsize=(6, 4))
    sns.heatmap(correlacao_pearson, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlação entre Novos Casos e Vacinação Diária")
    plt.show()

# Executar análises
descritiva()
probabilidade()
inferencia()
chi2()
serie_temporal()
correlacao()