import streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO

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
    st.header("Filtros disponíveis")
    st.markdown("Utilize os filtros abaixo para refinar os dados exibidos no dashboard:")

    selected_gender = st.multiselect(
        "Sexo",
        options=df["gender"].dropna().unique(),
        default=df["gender"].dropna().unique()
    )

    selected_country = st.multiselect(
        "País",
        options=df["country"].dropna().unique(),
        default=df["country"].dropna().unique()
    )

# Aplicar filtros
df_filtered = df[
    (df["gender"].isin(selected_gender)) &
    (df["country"].isin(selected_country))
]

# ===============================
# Layout com abas
# ===============================
st.title("Dashboard de Predição de Câncer Oral")

tabs = st.tabs(["Visão Geral", "Visualizações", "Mapa"])

# === Aba 1: Visão Geral ===
with tabs[0]:
    st.markdown("""
    Este painel interativo permite explorar dados relacionados ao câncer oral com base em fatores de risco,
    diagnóstico e taxas de sobrevivência.  
    Utilize os filtros na barra lateral para refinar os dados apresentados.
    """)
    st.dataframe(df_filtered.head(10))

# === Aba 2: Visualizações ===
with tabs[1]:
    st.subheader("Distribuição dos Diagnósticos de Câncer Oral")
    fig_diag = px.histogram(
        df_filtered,
        x="oral_cancer_diagnosis",
        color="oral_cancer_diagnosis",
        title="Diagnóstico",
        labels={"oral_cancer_diagnosis": "Diagnóstico"}
    )
    st.plotly_chart(fig_diag)

    html_diag = fig_diag.to_html().encode("utf-8")
    st.download_button(
        label="Baixar Gráfico de Diagnóstico (HTML)",
        data=html_diag,
        file_name="grafico_diagnostico.html",
        mime="text/html"
    )

    st.subheader("Distribuição por Estágio do Câncer")
    fig_stage = px.histogram(
        df_filtered,
        x="cancer_stage",
        color="cancer_stage",
        title="Estágios do Câncer Oral",
        labels={"cancer_stage": "Estágio"}
    )
    st.plotly_chart(fig_stage)

    html_stage = fig_stage.to_html().encode("utf-8")
    st.download_button(
        label="Baixar Gráfico de Estágios (HTML)",
        data=html_stage,
        file_name="grafico_estagios.html",
        mime="text/html"
    )

    st.subheader("Taxa de Sobrevivência em 5 Anos")
    fig_surv = px.box(
        df_filtered,
        x="gender",
        y="survival_rate_5_year_pct",
        color="gender",
        title="Taxa de Sobrevivência por Sexo",
        labels={"survival_rate_5_year_pct": "Taxa de Sobrevivência (%)"}
    )
    st.plotly_chart(fig_surv)

# === Aba 3: Mapa ===
with tabs[2]:
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
        template="plotly_white",
        projection="natural earth"
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


