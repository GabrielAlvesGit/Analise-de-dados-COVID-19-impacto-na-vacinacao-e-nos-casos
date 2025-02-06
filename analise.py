import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Carregar o dataframe
df = pd.read_csv("merged_filtered_data.csv")


def descritiva():
    # Estatística descritiva
    # --Média--
    media_novos_casos = df["new_cases"].mean()
    media_vacinacao = df["daily_vaccinations"].mean()
    media_mortes = df["new_deaths"].mean()

    # --Mediana--
    mediana_novos_casos = df["new_cases"].median()
    mediana_vacinacao = df["daily_vaccinations"].median()
    mediana_mortes = df["new_deaths"].median()

    # --Moda--
    moda_novos_casos = df["new_cases"].mode()[0]
    moda_vacinacao = df["daily_vaccinations"].mode()[0]
    moda_mortes = df["new_deaths"].mode()[0]

    # --Desvio padrão--
    desvio_padrao_novos_casos = df["new_cases"].std()
    desvio_padrao_vacinacao = df["daily_vaccinations"].std()
    desvio_padrao_mortes = df["new_deaths"].std()

    # Plotar gráficos
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # Gráfico de novos casos
    axs[0, 0].hist(df["new_cases"].dropna(), bins=30, color="blue", alpha=0.7)
    axs[0, 0].set_title("Distribuição de Novos Casos")
    axs[0, 0].set_xlabel("Novos Casos")
    axs[0, 0].set_ylabel("Frequência")
    axs[0, 0].axvline(
        media_novos_casos, color="red", linestyle="dashed", linewidth=1, label="Média"
    )
    axs[0, 0].axvline(
        mediana_novos_casos,
        color="green",
        linestyle="dashed",
        linewidth=1,
        label="Mediana",
    )
    axs[0, 0].axvline(
        moda_novos_casos, color="yellow", linestyle="dashed", linewidth=1, label="Moda"
    )
    axs[0, 0].axvline(
        media_novos_casos + desvio_padrao_novos_casos,
        color="purple",
        linestyle="dashed",
        linewidth=1,
        label="+1 Desvio Padrão",
    )
    axs[0, 0].axvline(
        media_novos_casos - desvio_padrao_novos_casos,
        color="purple",
        linestyle="dashed",
        linewidth=1,
        label="-1 Desvio Padrão",
    )
    axs[0, 0].legend()

    # Gráfico de vacinação diária
    axs[0, 1].hist(df["daily_vaccinations"].dropna(), bins=30, color="blue", alpha=0.7)
    axs[0, 1].set_title("Distribuição de Vacinação Diária")
    axs[0, 1].set_xlabel("Vacinações Diárias")
    axs[0, 1].set_ylabel("Frequência")
    axs[0, 1].axvline(
        media_vacinacao, color="red", linestyle="dashed", linewidth=1, label="Média"
    )
    axs[0, 1].axvline(
        mediana_vacinacao,
        color="green",
        linestyle="dashed",
        linewidth=1,
        label="Mediana",
    )
    axs[0, 1].axvline(
        moda_vacinacao, color="yellow", linestyle="dashed", linewidth=1, label="Moda"
    )
    axs[0, 1].axvline(
        media_vacinacao + desvio_padrao_vacinacao,
        color="purple",
        linestyle="dashed",
        linewidth=1,
        label="+1 Desvio Padrão",
    )
    axs[0, 1].axvline(
        media_vacinacao - desvio_padrao_vacinacao,
        color="purple",
        linestyle="dashed",
        linewidth=1,
        label="-1 Desvio Padrão",
    )
    axs[0, 1].legend()

    # Gráfico de novas mortes
    axs[0, 2].hist(df["new_deaths"].dropna(), bins=30, color="blue", alpha=0.7)
    axs[0, 2].set_title("Distribuição de Novas Mortes")
    axs[0, 2].set_xlabel("Novas Mortes")
    axs[0, 2].set_ylabel("Frequência")
    axs[0, 2].axvline(
        media_mortes, color="red", linestyle="dashed", linewidth=1, label="Média"
    )
    axs[0, 2].axvline(
        mediana_mortes, color="green", linestyle="dashed", linewidth=1, label="Mediana"
    )
    axs[0, 2].axvline(
        moda_mortes, color="yellow", linestyle="dashed", linewidth=1, label="Moda"
    )
    axs[0, 2].axvline(
        media_mortes + desvio_padrao_mortes,
        color="purple",
        linestyle="dashed",
        linewidth=1,
        label="+1 Desvio Padrão",
    )
    axs[0, 2].axvline(
        media_mortes - desvio_padrao_mortes,
        color="purple",
        linestyle="dashed",
        linewidth=1,
        label="-1 Desvio Padrão",
    )
    axs[0, 2].legend()

    # Boxplot de novos casos
    axs[1, 0].boxplot(df["new_cases"].dropna())
    axs[1, 0].set_title("Boxplot de Novos Casos")
    axs[1, 0].set_xlabel("Novos Casos")

    # Boxplot de vacinação diária
    axs[1, 1].boxplot(df["daily_vaccinations"].dropna())
    axs[1, 1].set_title("Boxplot de Vacinação Diária")
    axs[1, 1].set_xlabel("Vacinações Diárias")

    # Boxplot de novas mortes
    axs[1, 2].boxplot(df["new_deaths"].dropna())
    axs[1, 2].set_title("Boxplot de Novas Mortes")
    axs[1, 2].set_xlabel("Novas Mortes")

    plt.tight_layout()
    plt.show()

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
serie_temporal()
correlacao()