import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ===============================
# Configuração da Página
# ===============================
st.set_page_config(page_title="Dashboard Câncer Oral", layout="wide")

# ===============================
# Carregar e padronizar os dados
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("oral_cancer_prediction_dataset.csv")
    df.rename(columns={
        "survival_rate_(5-year_%)": "survival_rate_5_year_pct",
        "oral_cancer_(diagnosis)": "oral_cancer_diagnosis"
    }, inplace=True)
    return df

df = load_data()

# ===============================
# Treinar modelo preditivo simples
# ===============================
@st.cache_resource
def train_model(df):
    features = ["age", "tobacco_use", "alcohol_consumption", "hpv_infection", "gender"]
    df_model = df[features + ["oral_cancer_diagnosis"]].copy()

    le_dict = {}
    for col in features + ["oral_cancer_diagnosis"]:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        le_dict[col] = le

    X = df_model[features]
    y = df_model["oral_cancer_diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    return model, le_dict, acc

model, le_dict, acc = train_model(df)

# ===============================
# Filtros
# ===============================
st.title("Dashboard de Predição de Câncer Oral")
st.markdown("Este dashboard interativo permite explorar os dados relacionados aos fatores de risco e diagnósticos de câncer oral.")

st.sidebar.markdown("### Filtros de Dados")
selected_gender = st.sidebar.multiselect("Sexo", options=df["gender"].unique(), default=df["gender"].unique())
selected_country = st.sidebar.multiselect("País", options=df["country"].unique(), default=df["country"].unique())

df_filtered = df[(df["gender"].isin(selected_gender)) & (df["country"].isin(selected_country))]

# ===============================
# Métricas principais
# ===============================
st.subheader("Visão Geral dos Dados")
col1, col2, col3 = st.columns(3)
col1.metric("Total de Registros", f"{df_filtered.shape[0]:,}")
col2.metric("Países", df_filtered["country"].nunique())
col3.metric("Média de Idade", f"{df_filtered['age'].mean():.1f}")

# ===============================
# Gráficos
# ===============================
# Casos por País
st.subheader("Distribuição de Casos por País")
country_counts = df_filtered["country"].value_counts().reset_index()
country_counts.columns = ["country", "count"]
fig1 = px.bar(country_counts, x="country", y="count", color="count", title="Casos por País", template="plotly_white")
st.plotly_chart(fig1)

# Scatter: Idade vs Sobrevivência
st.subheader("Idade vs Taxa de Sobrevivência")
fig2 = px.scatter(
    df_filtered,
    x="age",
    y="survival_rate_5_year_pct",
    color="oral_cancer_diagnosis",
    hover_data=["country", "gender"],
    title="Idade vs Taxa de Sobrevivência (5 anos)",
    template="plotly_white"
)
st.plotly_chart(fig2)

# Mapa
st.subheader("Mapa Interativo: Casos por País")
fig3 = px.choropleth(
    country_counts,
    locations="country",
    locationmode="country names",
    color="count",
    color_continuous_scale="Blues",
    projection="natural earth",
    title="Distribuição de Casos por País",
    template="plotly_white"
)
st.plotly_chart(fig3)

# ===============================
# Predição
# ===============================
st.subheader("🔍 Predição de Câncer Oral")
st.markdown("Preencha os campos abaixo para prever se há risco de diagnóstico de câncer oral:")

input_age = st.slider("Idade", 10, 100, 50)
input_gender = st.selectbox("Sexo", df["gender"].unique())
input_tobacco = st.selectbox("Uso de Tabaco", df["tobacco_use"].unique())
input_alcohol = st.selectbox("Consumo de Álcool", df["alcohol_consumption"].unique())
input_hpv = st.selectbox("Infecção por HPV", df["hpv_infection"].unique())

if st.button("🔎 Prever"):
    input_df = pd.DataFrame({
        "age": [input_age],
        "gender": [input_gender],
        "tobacco_use": [input_tobacco],
        "alcohol_consumption": [input_alcohol],
        "hpv_infection": [input_hpv]
    })

    for col in input_df.columns:
        input_df[col] = le_dict[col].transform(input_df[col])

    prediction = model.predict(input_df)[0]
    resultado = le_dict["oral_cancer_diagnosis"].inverse_transform([prediction])[0]

    st.success(f"🎯 Diagnóstico Previsto: **{resultado}**")
    st.info(f"Acurácia do modelo: {acc*100:.2f}%")


