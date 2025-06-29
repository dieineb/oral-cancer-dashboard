import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def load_data():
    # CSV já com colunas padronizadas manualmente
    return pd.read_csv("oral_cancer_prediction_dataset.csv")

df = load_data()

st.title("Dashboard de Predição de Câncer Oral")
st.markdown("Este dashboard interativo permite explorar os dados relacionados aos fatores de risco e diagnósticos de câncer oral.")

# Sidebar
st.sidebar.header("Filtros")
selected_gender = st.sidebar.multiselect("Sexo", options=df["gender"].unique(), default=df["gender"].unique())
selected_country = st.sidebar.multiselect("País", options=df["country"].unique(), default=df["country"].unique())

df_filtered = df[
    (df["gender"].isin(selected_gender)) &
    (df["country"].isin(selected_country))
]

# Métricas
st.subheader("Visão Geral dos Dados")
col1, col2, col3 = st.columns(3)
col1.metric("Total de Registros", f"{df_filtered.shape[0]:,}")
col2.metric("Países", df_filtered["country"].nunique())
col3.metric("Média de Idade", f"{df_filtered['age'].mean():.1f}")

# Gráfico: Casos por País
st.subheader("Distribuição de Casos por País")
country_counts = df_filtered["country"].value_counts().reset_index()
country_counts.columns = ["country", "count"]
fig1 = px.bar(country_counts, x="country", y="count", title="Casos por País", color="count")
st.plotly_chart(fig1)

# Gráfico: Idade vs Sobrevivência
st.subheader("Idade vs Taxa de Sobrevivência")
fig2 = px.scatter(
    df_filtered,
    x="age",
    y="survival_rate_5_year_%",
    color="oral_cancer_(diagnosis)",
    hover_data=["country", "gender"],
    title="Idade vs Taxa de Sobrevivência (5 anos)"
)
st.plotly_chart(fig2)

# Mapa: Casos por País
st.subheader("Mapa Interativo: Casos por País")
fig3 = px.choropleth(
    country_counts,
    locations="country",
    locationmode="country names",
    color="count",
    color_continuous_scale="Reds",
    title="Distribuição de Casos por País"
)
st.plotly_chart(fig3)

