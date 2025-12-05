import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import plotly.express as px
from phik import phik_matrix

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "price_predict_model.pkl"      
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"

# Загрузка модели
@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURE_NAMES_PATH, 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names

MODEL, FEATURE_NAMES = load_model()


# Подготовка признаков
def prepare_features(df, feature_names):
    df_proc = df.copy()
    for col in feature_names:
        if col not in df_proc.columns:
            df_proc[col] = 0
        if df_proc[col].dtype == "object":
            df_proc[col] = df_proc[col].astype(str)
    return df_proc[feature_names]



# Основной интерфейс
st.title("🚗 Car Price Prediction Dashboard")
uploaded_file = st.file_uploader("Загрузите CSV файл с данными", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Файл успешно загружен!")
    st.write(df.head())
else:
    st.info("Загрузите CSV-файл для EDA, визуализаций и предсказаний.")
    st.stop()


# EDA и визуализации
st.header("📊 Exploratory Data Analysis")
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# Correlation (Pearson)
st.subheader("📌 Корреляция Pearson")
corr = df[numeric_cols].corr()
fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Pearson Correlation")
st.plotly_chart(fig, use_container_width=True)

# Phik correlation
st.subheader("📌 Phik-корреляция")
try:
    phik_matrix = df.phik_matrix(interval_cols=numeric_cols)
    fig = px.imshow(phik_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Phik Correlation")
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Не удалось посчитать phik: {e}")

# Histogram для numeric
st.subheader("📌 Распределения числовых признаков")
for col in numeric_cols:
    fig = px.histogram(df, x=col, nbins=30, title=f"Распределение {col}")
    st.plotly_chart(fig, use_container_width=True)

# Scatter: Selling Price vs Max Power/Torque
if {"max_power", "torque", "selling_price", "engine"}.issubset(df.columns):
    st.subheader("📌 Selling Price vs Max Power/Torque")
    fig = px.scatter(df, x="max_power", y="selling_price", color="engine",
                     log_y=True, title="Selling Price vs Max Power")
    st.plotly_chart(fig, use_container_width=True)
    
    fig2 = px.scatter(df, x="torque", y="selling_price", color="engine",
                      log_y=True, title="Selling Price vs Torque")
    st.plotly_chart(fig2, use_container_width=True)

# Boxplot: fuel
if {"fuel", "selling_price"}.issubset(df.columns):
    st.subheader("📌 Boxplot по типу топлива")
    fig = px.box(df, x="fuel", y="selling_price", title="Цена по типу топлива", color="fuel")
    st.plotly_chart(fig, use_container_width=True)

# Boxenplot: seller_type
if {"seller_type", "selling_price"}.issubset(df.columns):
    st.subheader("📌 Boxenplot по виду продавца")
    fig = px.box(df, x="seller_type", y="selling_price", title="Цена по виду продавца", color="seller_type")
    st.plotly_chart(fig, use_container_width=True)


# Предсказания модели для CSV
st.header("🔮 Предсказания модели")

try:
    features = prepare_features(df, FEATURE_NAMES)
    predictions = MODEL.predict(features)
    df["predicted_price"] = predictions

    st.success("Предсказания успешно рассчитаны!")
    st.write(df[["predicted_price"]].head())

    fig = px.histogram(df, x="predicted_price", nbins=30, title="Распределение предсказанных цен")
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Ошибка при предсказании: {e}")

st.subheader("📌 Pairplot (интерактивный)")

st.subheader("📈 Распределение предсказанных цен (KDE)")

if "predicted_price" in df.columns:
    fig = px.histogram(
        df, 
        x="predicted_price", 
        nbins=50, 
        histnorm='density',
        title="KDE распределение предсказанных цен",
        marginal="rug"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Сначала нужно сделать предсказания, чтобы построить распределение.")

# Ограничим количество признаков для наглядности
pairplot_cols = numeric_cols[:10]  # первые 10 числовых признаков
if len(pairplot_cols) >= 2:
    fig = px.scatter_matrix(
        df[pairplot_cols],
        dimensions=pairplot_cols,
        color=df[categorical_cols[0]] if categorical_cols else None,
        title="Pairplot числовых признаков",
        height=800
    )
    fig.update_traces(diagonal_visible=False)  # скрыть диагональные графики
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Слишком мало числовых признаков для pairplot.")

# Форма ручного ввода признаков
st.header("✍️ Предсказание цены для одного объекта")

with st.form("form_manual"):
    input_data = {}
    for col in FEATURE_NAMES:
        if col in categorical_cols:
            values = df[col].astype(str).unique().tolist()
            input_data[col] = st.selectbox(col, values)
        else:
            input_data[col] = st.number_input(col, value=float(df[col].median()))
    submit = st.form_submit_button("Предсказать цену")

if submit:
    input_df = pd.DataFrame([input_data])
    input_ready = prepare_features(input_df, FEATURE_NAMES)
    price_pred = MODEL.predict(input_ready)[0]
    st.success(f"Предсказанная цена: **{price_pred:,.2f}**")


# Визуализация весов модели
st.header("📐 Веса обученной модели")

if hasattr(MODEL, "coef_"):
    coefs = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "coef": MODEL.coef_
    }).sort_values("coef", key=abs, ascending=False)

    st.dataframe(coefs)

    fig = px.bar(coefs, x="coef", y="feature", orientation="h", title="Важность признаков (веса модели)")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("У модели нет атрибута coef_ – невозможно показать веса.")
