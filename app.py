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
'''
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
'''

# ===============================
# Filtros laterais 
# ===============================
with st.sidebar:
    st.header("Filtros")
    st.markdown("Selecione os critérios para refinar os dados:")

    # Filtro por sexo
    selected_gender = st.selectbox(
        "Sexo",
        options=["Todos"] + sorted(df["gender"].dropna().unique().tolist())
    )

    # Filtro por país
    selected_country = st.selectbox(
        "País",
        options=["Todos"] + sorted(df["country"].dropna().unique().tolist())
    )

    # Novo: Filtro por faixa etária (faixas fixas)
    age_ranges = {
        "20-40": (20, 40),
        "41-60": (41, 60),
        "61-80": (61, 80),
        "81+": (81, 120)
    }
    selected_age_ranges = st.multiselect(
        "Faixa Etária",
        options=list(age_ranges.keys()),
        default=list(age_ranges.keys())
    )

    # Novo: Filtro por estágio do câncer
    available_stages = df["cancer_stage"].dropna().unique().tolist()
    selected_stages = st.multiselect(
        "Estágio do Câncer",
        options=available_stages,
        default=available_stages
    )
# ===============================
# Aplicar filtros
# ===============================
df_filtered = df.copy()

# Filtro por sexo
if selected_gender != "Todos":
    df_filtered = df_filtered[df_filtered["gender"] == selected_gender]

# Filtro por país
if selected_country != "Todos":
    df_filtered = df_filtered[df_filtered["country"] == selected_country]

# Filtro por estágio do câncer
if selected_stages:
    df_filtered = df_filtered[df_filtered["cancer_stage"].isin(selected_stages)]

# Filtro por faixa etária
if selected_age_ranges:
    age_mask = pd.Series(False, index=df_filtered.index)
    for faixa in selected_age_ranges:
        min_age, max_age = age_ranges[faixa]
        age_mask |= (df_filtered["age"] >= min_age) & (df_filtered["age"] <= max_age)
    df_filtered = df_filtered[age_mask]

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


