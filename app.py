
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

st.set_page_config(page_title="Dashboard - Câncer Oral", layout="wide")

st.title("🧪 Dashboard: Predição de Câncer Oral")

@st.cache_data
def load_data():
    df = pd.read_csv("oral_cancer_prediction_dataset.csv")
    return df

df = load_data()

st.sidebar.header("🔍 Filtros")
selected_gender = st.sidebar.multiselect("Sexo", options=df["Gender"].dropna().unique(), default=df["Gender"].dropna().unique())
selected_country = st.sidebar.multiselect("País", options=df["Country"].dropna().unique(), default=df["Country"].dropna().unique())

df_filtered = df[(df["Gender"].isin(selected_gender)) & (df["Country"].isin(selected_country))]

st.subheader("📊 Amostra dos Dados")
st.dataframe(df_filtered.head())

st.subheader("📈 Distribuição de Idade por Diagnóstico")
fig1 = px.histogram(df_filtered, x="Age", color="Oral Cancer (Diagnosis)", barmode="overlay", nbins=30)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("🚬 Uso de Tabaco e Diagnóstico")
fig2 = px.histogram(df_filtered, x="Tobacco Use", color="Oral Cancer (Diagnosis)")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("🗺️ Mapa Interativo por País")
map_data = df.groupby("Country")["Oral Cancer (Diagnosis)"].value_counts().unstack().fillna(0)
map_data["Total"] = map_data.sum(axis=1)
map_data["% Câncer Oral"] = (map_data[1] / map_data["Total"]) * 100
map_data.reset_index(inplace=True)

fig_map = px.choropleth(
    map_data,
    locations="Country",
    locationmode="country names",
    color="% Câncer Oral",
    hover_name="Country",
    color_continuous_scale="Reds",
    title="% de Casos Positivos de Câncer Oral por País"
)
st.plotly_chart(fig_map, use_container_width=True)

st.subheader("🤖 Modelo de Regressão Logística")

if st.checkbox("Rodar modelo preditivo"):
    features = ["Age", "Tobacco Use", "Alcohol Consumption", "HPV Infection", "Betel Quid Use"]
    model_df = df_filtered[features + ["Oral Cancer (Diagnosis)"]].dropna()
    model_df = pd.get_dummies(model_df, drop_first=True)

    X = model_df.drop("Oral Cancer (Diagnosis)", axis=1)
    y = model_df["Oral Cancer (Diagnosis)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.text("Relatório de Classificação:")
    st.text(classification_report(y_test, y_pred))
