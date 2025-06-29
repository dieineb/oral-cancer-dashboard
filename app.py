
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

st.markdown("---")

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
    template="plotly_white"
)
st.plotly_chart(fig_diag)

st.subheader("Distribuição por Estágio do Câncer")
fig_stage = px.histogram(
    df_filtered,
    x="cancer_stage",
    color="cancer_stage",
    title="Estágios do Câncer Oral",
    labels={"cancer_stage": "Estágio"},
    template="ggplot2"
)
st.plotly_chart(fig_stage)

st.subheader("Taxa de Sobrevivência por Sexo")
fig_surv = px.violin(
    df_filtered,
    y="survival_rate_5_year_pct",
    x="gender",
    box=True,
    points="all",
    color="gender",
    title="Taxa de Sobrevivência por Sexo",
    labels={"survival_rate_5_year_pct": "Taxa de Sobrevivência (%)"},
    template="seaborn"
)
st.plotly_chart(fig_surv)

st.subheader("Taxa de Sobrevivência por Idade")
fig_age = px.scatter(
    df_filtered,
    x="age",
    y="survival_rate_5_year_pct",
    color="gender",
    title="Taxa de Sobrevivência vs Idade",
    labels={"survival_rate_5_year_pct": "Taxa de Sobrevivência (%)"},
    trendline="ols",
    template="plotly_dark"
)
st.plotly_chart(fig_age)

st.subheader("Taxa de Sobrevivência por Estágio do Câncer")
fig_surv_stage = px.box(
    df_filtered,
    x="cancer_stage",
    y="survival_rate_5_year_pct",
    color="cancer_stage",
    title="Taxa de Sobrevivência por Estágio do Câncer",
    labels={"survival_rate_5_year_pct": "Taxa de Sobrevivência (%)"},
    template="simple_white"
)
st.plotly_chart(fig_surv_stage)

st.subheader("Distribuição Geográfica dos Casos")
country_counts = df_filtered["country"].value_counts().reset_index()
country_counts.columns = ["country", "cases"]

fig_map = px.choropleth(
    country_counts,
    locations="country",
    locationmode="country names",
    color="cases",
    hover_name="country",
    color_continuous_scale="Viridis",
    title="Distribuição Global de Casos de Câncer Oral",
    template="plotly",
    projection="natural earth"
)
fig_map.update_geos(
    showcoastlines=True, coastlinecolor="gray",
    showland=True, landcolor="white",
    showlakes=True, lakecolor="lightblue"
)
st.plotly_chart(fig_map, use_container_width=True)



