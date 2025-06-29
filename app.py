import streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

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
st.sidebar.title("Filtros")
selected_gender = st.sidebar.multiselect("Sexo", options=df["gender"].unique(), default=df["gender"].unique())
selected_country = st.sidebar.multiselect("País", options=df["country"].unique(), default=df["country"].unique())

# Aplicar filtros
df_filtered = df[
    (df["gender"].isin(selected_gender)) &
    (df["country"].isin(selected_country))
]

# ===============================
# Layout com abas
# ===============================
st.title("Dashboard de Predição de Câncer Oral")

tabs = st.tabs(["Visão Geral", "Visualizações", "Mapa", "Modelo Preditivo"])

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
        color_continuous_scale="blues",
        title="Casos por País",
        template="plotly_dark",
        projection="natural earth"
    )

    fig_map.update_geos(showcoastlines=True, coastlinecolor="white", showland=True, landcolor="rgb(240,240,240)")
    fig_map.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

# === Aba 4: Modelo Preditivo ===
with tabs[3]:
    st.subheader("Modelo Preditivo - Regressão Logística")

    df_model = df.copy()
    df_model["oral_cancer_diagnosis"] = df_model["oral_cancer_diagnosis"].map({"Healthy": 0, "Cancer": 1})

    features = ["age", "tobacco_use", "alcohol_consumption", "hpv_infection"]
    X = df_model[features]
    y = df_model["oral_cancer_diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=False)
    st.text(report)

    # Download do relatório
    report_buffer = StringIO()
    report_buffer.write(report)
    st.download_button(
        label="Baixar Relatório do Modelo (TXT)",
        data=report_buffer.getvalue(),
        file_name="relatorio_modelo.txt",
        mime="text/plain"
    )


