import streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO

@st.cache_data
def load_data():
    df = pd.read_csv("oral_cancer_prediction_dataset.csv")
    df.rename(columns={
        "survival_rate_5-year_pct": "survival_rate_5_year_pct"
    }, inplace=True)
    return df

df = load_data()

# ===============================
# Filtros laterais com multiselect
# ===============================
with st.sidebar:
    st.header("Filtros")
    st.markdown("Selecione os critérios para refinar os dados:")

    selected_gender = st.multiselect(
        "Sexo",
        options=sorted(df["gender"].dropna().unique()),
        default=sorted(df["gender"].dropna().unique())
    )

    selected_country = st.multiselect(
        "País",
        options=sorted(df["country"].dropna().unique()),
        default=sorted(df["country"].dropna().unique())
    )

# Aplicar filtros
df_filtered = df[
    (df["gender"].isin(selected_gender)) &
    (df["country"].isin(selected_country))
]

# ===============================
# Título e métricas
# ===============================
st.title("Dashboard de Predição de Câncer Oral")

total_registros = df_filtered.shape[0]
media_idade = df_filtered["age"].mean()
num_paises = df_filtered["country"].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("Total de Registros", total_registros)
col2.metric("Idade Média", f"{media_idade:.1f} anos")
col3.metric("Países Representados", num_paises)

# ===============================
# Abas
# ===============================
tabs = st.tabs(["Visão Geral", "Visualizações", "Mapa"])

# === Aba 1: Visão Geral ===
with tabs[0]:
    st.markdown("Esta aba apresenta um resumo estatístico dos dados filtrados.")

    st.subheader("Distribuição por Sexo")
    sexo_counts = df_filtered["gender"].value_counts().reset_index()
    sexo_counts.columns = ["Sexo", "Total"]
    fig_sexo = px.pie(sexo_counts, names="Sexo", values="Total", hole=0.4)
    st.plotly_chart(fig_sexo)

    st.subheader("Diagnóstico (Healthy x Cancer)")
    diag_counts = df_filtered["oral_cancer_diagnosis"].value_counts().reset_index()
    diag_counts.columns = ["Diagnóstico", "Total"]
    fig_diag_resumo = px.bar(diag_counts, x="Diagnóstico", y="Total", color="Diagnóstico")
    st.plotly_chart(fig_diag_resumo)

    st.subheader("Estágios do Câncer Oral")
    stage_counts = df_filtered["cancer_stage"].value_counts().reset_index()
    stage_counts.columns = ["Estágio", "Casos"]
    fig_stage_resumo = px.bar(stage_counts, x="Estágio", y="Casos", color="Estágio")
    st.plotly_chart(fig_stage_resumo)

    with st.expander("Visualizar tabela de dados filtrados"):
        st.dataframe(df_filtered.reset_index(drop=True))

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

    st.subheader("Distribuição dos Fatores de Risco")
    fatores_binarios = [
        "tobacco_use", "alcohol_consumption", "hpv_infection",
        "betel_quid_use", "chronic_sun_exposure", "poor_oral_hygiene",
        "family_history_of_cancer", "compromised_immune_system"
    ]
    percentuais = df_filtered[fatores_binarios].mean().sort_values(ascending=False) * 100
    df_risco = pd.DataFrame({
        "Fator de Risco": percentuais.index.str.replace("_", " ").str.title(),
        "Porcentagem": percentuais.values
    })
    fig_risco = px.bar(
        df_risco,
        x="Fator de Risco",
        y="Porcentagem",
        title="Prevalência dos Fatores de Risco",
        labels={"Porcentagem": "% de Presença"}
    )
    fig_risco.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_risco)

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
        color_continuous_scale="Tealgrn",
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


