
import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def load_data():
    df = pd.read_csv("oral_cancer_prediction_dataset.csv")
    df.rename(columns={
        "survival_rate_5-year_pct": "survival_rate_5_year_pct"
    }, inplace=True)
    return df

df = load_data()

with st.sidebar:
    st.header("Filtros")
    st.markdown("Selecione os critérios para refinar os dados:")

    genders = sorted(df["gender"].dropna().unique().tolist())
    countries = sorted(df["country"].dropna().unique().tolist())

    selected_gender = st.multiselect("Sexo", options=genders, default=genders)
    selected_country = st.multiselect("País", options=countries, default=countries)

df_filtered = df[
    (df["gender"].isin(selected_gender)) &
    (df["country"].isin(selected_country))
]

st.title("Dashboard de Predição de Câncer Oral")

total_registros = df_filtered.shape[0]
media_idade = df_filtered["age"].mean()
num_paises = df_filtered["country"].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("Total de Registros", total_registros)
col2.metric("Idade Média", f"{media_idade:.1f} anos")
col3.metric("Países Representados", num_paises)

tabs = st.tabs(["Visão Geral", "Visualizações", "Mapa"])

with tabs[0]:
    if df_filtered.empty:
        st.warning("Nenhum dado disponível com os filtros atuais.")
    else:
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

with tabs[2]:
    st.subheader("Distribuição Geográfica dos Casos")

    if df_filtered.empty:
        st.warning("Nenhum dado disponível com os filtros atuais.")
    else:
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
            showland=True,
            landcolor="whitesmoke",
            showocean=True,
            oceancolor="lightblue"
        )

        fig_map.update_layout(
            title_font_size=20,
            margin=dict(r=0, t=60, l=0, b=0)
        )

        st.plotly_chart(fig_map, use_container_width=True)


