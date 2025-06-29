import streamlit as st
import pandas as pd
import plotly.express as px

# ===============================
# Carregar e padronizar os dados
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("oral_cancer_prediction_dataset.csv")
    df.rename(columns={
        "survival_rate_5-year_pct": "survival_rate_5_year_pct"
    }, inplace=True)
    return df

df = load_data()

# ===============================
# Filtros laterais 
# ===============================
with st.sidebar:
    st.header("Filtros")
    st.markdown("Selecione os critérios para refinar os dados:")

    selected_gender = st.selectbox(
        "Sexo",
        options=["Todos"] + sorted(df["gender"].dropna().unique().tolist())
    )
    
    selected_country = st.selectbox(
        "País",
        options=["Todos"] + sorted(df["country"].dropna().unique().tolist())
    )
    
# Aplicar filtros
df_filtered = df.copy()
if selected_gender != "Todos":
    df_filtered = df_filtered[df_filtered["gender"] == selected_gender]
if selected_country != "Todos":
    df_filtered = df_filtered[df_filtered["country"] == selected_country]

# ===============================
# Título
# ===============================
st.title("Dashboard de Predição de Câncer Oral")

# ===============================
# Exibir métricas no topo
# ===============================
total_registros = df_filtered.shape[0]
media_idade = df_filtered["age"].mean()
num_paises = df_filtered["country"].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("Total de Registros", total_registros)
col2.metric("Idade Média", f"{media_idade:.1f} anos")
col3.metric("Países Representados", num_paises)

# ===============================
# Visualizações
# ===============================
st.subheader("Distribuição dos Diagnósticos de Câncer Oral")
fig_diag = px.histogram(
    df_filtered,
    x="oral_cancer_diagnosis",
    color="oral_cancer_diagnosis",
    title="Diagnóstico",
    labels={"oral_cancer_diagnosis": "Diagnóstico"},
    template="simple_white"
)
st.plotly_chart(fig_diag)

st.subheader("Distribuição por Estágio do Câncer")
fig_stage = px.histogram(
    df_filtered,
    x="cancer_stage",
    color="cancer_stage",
    title="Estágios do Câncer Oral",
    labels={"cancer_stage": "Estágio"},
    template="simple_white"
)
st.plotly_chart(fig_stage)

st.subheader("Taxa de Sobrevivência por Sexo")
fig_surv = px.box(
    df_filtered,
    x="gender",
    y="survival_rate_5_year_pct",
    color="gender",
    title="Taxa de Sobrevivência por Sexo",
    labels={"survival_rate_5_year_pct": "Taxa de Sobrevivência (%)"},
    template="simple_white"
)
st.plotly_chart(fig_surv)

st.subheader("Taxa de Sobrevivência por Idade")
df_age_plot = df_filtered[["age", "survival_rate_5_year_pct", "cancer_stage"]].dropna()
df_age_plot = df_age_plot[df_age_plot["age"].apply(lambda x: isinstance(x, (int, float)))]

fig_age = px.scatter(
    df_age_plot,
    x="age",
    y="survival_rate_5_year_pct",
    color="cancer_stage",
    title="Taxa de Sobrevivência por Idade",
    labels={"age": "Idade", "survival_rate_5_year_pct": "Taxa de Sobrevivência (%)"},
    template="simple_white"
)
st.plotly_chart(fig_age)

# ===============================
# Mapa
# ===============================
st.subheader("Distribuição Geográfica dos Casos")

country_counts = df_filtered["country"].value_counts().reset_index()
country_counts.columns = ["country", "cases"]

fig_map = px.choropleth(
    country_counts,
    locations="country",
    locationmode="country names",
    color="cases",
    hover_name="country",
    color_continuous_scale="Blues",
    template="simple_white",
    projection="natural earth",
    title="Distribuição Global de Casos de Câncer Oral",
    color_continuous_midpoint=country_counts["cases"].mean()
)

fig_map.update_geos(
    showcoastlines=True,
    coastlinecolor="gray",
    showland=True,
    landcolor="rgb(245, 245, 245)",
    showlakes=True,
    lakecolor="lightblue"
)

fig_map.update_layout(
    title_font_size=20,
    margin=dict(r=0, t=60, l=0, b=0)
)

st.plotly_chart(fig_map, use_container_width=True)
st.subheader("Análise por Grupos de Fatores de Risco")

def plot_group(title, factor_list, color):
    # Contagem absoluta e percentual
    risk_counts = df_filtered[factor_list].apply(lambda col: (col == "Yes").sum())
    percentuais = risk_counts / len(df_filtered) * 100

    df_group = pd.DataFrame({
        "Fator de Risco": risk_counts.index.str.replace('_', ' ').str.title(),
        "Casos com Presença": risk_counts.values,
        "Percentual (%)": percentuais.values
    })

    # Gráfico com hover visível e legível
    fig = px.bar(
        df_group,
        x="Casos com Presença",
        y="Fator de Risco",
        orientation="h",
        color_discrete_sequence=[color],
        template="simple_white",
        title=title,
        hover_data={
            "Casos com Presença": True,
            "Percentual (%)": ':.1f',
            "Fator de Risco": False
        }
    )

    fig.update_layout(
        yaxis=dict(categoryorder='total ascending'),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.95)",  # fundo claro
            font=dict(color="black", size=12)
        ),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

# Estilo de Vida
plot_group("Estilo de Vida", [
    "tobacco_use", "alcohol_consumption", "betel_quid_use", "diet_fruits_vegetables_intake"
], "#636EFA")

# Biológicos
plot_group("Fatores Biológicos", [
    "hpv_infection", "compromised_immune_system", "family_history_of_cancer"
], "#EF553B")

# Ambientais
plot_group("Exposição Ambiental", [
    "chronic_sun_exposure", "poor_oral_hygiene"
], "#00CC96")

# Clínicos
plot_group("Sinais Clínicos", [
    "oral_lesions", "unexplained_bleeding", "difficulty_swallowing", "white_or_red_patches_in_mouth"
], "#AB63FA")

# ===============================
# Análise Econômica e Clínica dos Tratamentos
# ===============================
st.subheader("Análise Econômica e Clínica dos Tratamentos")

#Distribuição dos Tipos de Tratamento
fig_tt = px.histogram(
    df_filtered,
    x="treatment_type",
    color="treatment_type",
    title="Tipos de Tratamento",
    labels={"treatment_type": "Tipo de Tratamento"},
    template="simple_white"
)
st.plotly_chart(fig_tt)

# Custo Médio por Tipo de Tratamento
df_tt_cost = df_filtered.groupby("treatment_type")["cost_of_treatment_usd"].mean().reset_index()

fig_tt_cost = px.bar(
    df_tt_cost,
    x="treatment_type",
    y="cost_of_treatment_usd",
    color="treatment_type",
    title="Custo Médio (USD) por Tipo de Tratamento",
    labels={
        "cost_of_treatment_usd": "Custo Médio (USD)",
        "treatment_type": "Tipo de Tratamento"
    },
    template="simple_white"
)
st.plotly_chart(fig_tt_cost)

st.subheader("Custo Médio por Faixa Etária")

# Criar faixas etárias
df_age_cost = df_filtered.dropna(subset=["age", "cost_of_treatment_usd"])
df_age_cost["faixa_etaria"] = pd.cut(df_age_cost["age"], bins=[0, 30, 45, 60, 75, 90, 120],
                                     labels=["0-30", "31-45", "46-60", "61-75", "76-90", "91+"])

df_avg_cost_age = df_age_cost.groupby("faixa_etaria")["cost_of_treatment_usd"].mean().reset_index()

fig_line_age_cost = px.line(
    df_avg_cost_age,
    x="faixa_etaria",
    y="cost_of_treatment_usd",
    title="Custo Médio por Faixa Etária",
    labels={
        "faixa_etaria": "Faixa Etária",
        "cost_of_treatment_usd": "Custo Médio (USD)"
    },
    markers=True,
    template="simple_white"
)
st.plotly_chart(fig_line_age_cost)

# ===============================
# Informações sobre o Dataset
# ===============================
st.markdown("---")
st.markdown("### Informações sobre o Dataset")

st.markdown("""
- **Nome:** Oral Cancer Prediction Dataset  
- **Fonte:** Kaggle ([link](https://www.kaggle.com/datasets/ankushpanday2/oral-cancer-prediction-dataset))  
- **Cobertura Temporal:**  
  - Início: **08 de agosto de 2018**  
  - Fim: **01 de fevereiro de 2024**  
- **Cobertura Geográfica:** Mundial 🌍  
- **Descrição:** Conjunto de dados contendo informações clínicas, demográficas e econômicas relacionadas ao câncer oral. As variáveis incluem:
  - Idade, sexo, país
  - Estágio do câncer e tipo de tratamento
  - Custo do tratamento em dólares (USD)
  - Taxa de sobrevivência em 5 anos (%)
  - Impacto econômico (dias de trabalho perdidos por ano)

- **Objetivo:** Permitir análises exploratórias e preditivas voltadas à compreensão dos fatores de risco, impacto econômico e desfechos clínicos do câncer oral.
""")
