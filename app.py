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
# Layout com abas
# ===============================
tabs = st.tabs(["Visão Geral", "Visualizações", "Mapa"])

# === Aba 1: Visão Geral ===
with tabs[0]:
    st.markdown(
        "Este painel permite explorar dados relacionados ao câncer oral com base em fatores de risco, "
        "diagnóstico e taxas de sobrevivência. Os dados podem ser filtrados pela barra lateral."
    )
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


        st.plotly_chart(fig_map, use_container_width=True)


