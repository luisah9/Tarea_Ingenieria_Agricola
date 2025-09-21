"""
Streamlit app adaptada para ejecución en Google Colab (con ngrok).
Esta versión selecciona por defecto 'label' como target si existe en el CSV
y selecciona automáticamente todas las features (excepto el target).
Guarda este archivo como streamlit_app_colab.py y súbelo a Colab junto con datos.csv.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from io import BytesIO
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_squared_error

st.set_page_config(page_title="Explorador & ML (Colab)", layout="wide")

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    default_path = os.path.join(os.getcwd(), "datos.csv")
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    return pd.DataFrame()

def summary_dataframe(df):
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_unique": df.nunique(),
        "n_missing": df.isnull().sum(),
    })

def get_feature_names(preprocessor, numeric_cols, categorical_cols):
    feat_names = []
    feat_names += numeric_cols
    try:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        ohe_names = ohe.get_feature_names_out(categorical_cols).tolist()
        feat_names += ohe_names
    except Exception:
        feat_names += categorical_cols
    return feat_names

st.title("Explorador de datos y entrenamiento rápido (Colab-friendly)")
st.sidebar.header("Configuración")

uploaded_file = st.sidebar.file_uploader("Sube un CSV (opcional) — normalmente 'datos.csv'", type=["csv"])
df = load_data(uploaded_file)

if df.empty:
    st.warning("No se ha cargado ningún dataset. Suba 'datos.csv' o seleccione un CSV en la barra lateral.")
    st.stop()

st.sidebar.markdown(f"*Filas:* {df.shape[0]}  \n*Columnas:* {df.shape[1]}")

if df.shape[0] < 50:
    st.info("Nota: el dataset tiene menos de 50 filas; los resultados de modelado pueden no ser representativos.")

tab1, tab2, tab3, tab4 = st.tabs(["Datos", "Visualizaciones", "Filtrar & Descargar", "Modelado"])

with tab1:
    st.header("Vista rápida del dataset")
    st.write("Resumen de columnas:")
    st.dataframe(summary_dataframe(df))
    st.write("Primeros registros:")
    st.dataframe(df.head(20))
    if st.checkbox("Mostrar información completa (info())"):
        import io
        buf = io.StringIO()
        df.info(buf=buf)
        st.text(buf.getvalue())

with tab2:
    st.header("Visualizaciones básicas")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    st.subheader("Histogramas")
    sel_hist = st.multiselect("Seleccionar columnas (numéricas) para histogramas", numeric_cols, default=(numeric_cols[:2] if len(numeric_cols)>=2 else numeric_cols))
    for col in sel_hist:
        fig, ax = plt.subplots()
        df[col].hist(ax=ax)
        ax.set_title(f"Histograma — {col}")
        st.pyplot(fig)

    st.subheader("Scatter (x vs y)")
    if len(numeric_cols) >= 2:
        x_col = st.selectbox("Eje X", numeric_cols, index=0)
        y_col = st.selectbox("Eje Y", numeric_cols, index=1 if len(numeric_cols)>1 else 0)
        fig, ax = plt.subplots()
        ax.scatter(df[x_col], df[y_col], s=10)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}")
        st.pyplot(fig)
    else:
        st.info("Se requieren al menos 2 columnas numéricas para scatter.")

    st.subheader("Mapa de correlación (numéricas)")
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(6, 4))
        cax = ax.matshow(corr)
        fig.colorbar(cax)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)
        st.pyplot(fig)
    else:
        st.info("No hay suficientes columnas numéricas para calcular correlación.")

with tab3:
    st.header("Filtrado interactivo y descarga")
    st.write("Selecciona filtros por columnas (se genera automáticamente)")
    df_filtered = df.copy()
    with st.form("filters_form"):
        for col in df.columns:
            if df[col].dtype.kind in 'biufc':
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                lo, hi = st.slider(f"{col} (rango)", min_val, max_val, (min_val, max_val))
                df_filtered = df_filtered[(df_filtered[col] >= lo) & (df_filtered[col] <= hi)]
            else:
                vals = df[col].dropna().unique().tolist()
                sel = st.multiselect(f"{col} (valores)", vals, default=vals[:5])
                if sel:
                    df_filtered = df_filtered[df_filtered[col].isin(sel)]
        submitted = st.form_submit_button("Aplicar filtros")
    st.write(f"Filtrado: {df_filtered.shape[0]} filas resultantes")
    st.dataframe(df_filtered.head(50))
    csv_bytes = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button("Descargar subset filtrado (CSV)", data=csv_bytes, file_name="subset_filtrado.csv", mime="text/csv")

with tab4:
    st.header("Entrenamiento rápido de modelo")
    st.write("Selecciona la columna objetivo (target). Esta versión intentará seleccionar 'label' por defecto y todas las features automáticamente.")
    cols = df.columns.tolist()

    default_idx = 0
    try:
        idx_in_cols = cols.index('label')
        default_idx = idx_in_cols + 1
    except ValueError:
        default_idx = 0

    target = st.selectbox("Columna objetivo (target)", options=["--NINGUNO--"] + cols, index=default_idx)
    if target and target != "--NINGUNO--":
        target_series = df[target]
        n_unique = target_series.nunique(dropna=True)
        is_numeric_target = pd.api.types.is_numeric_dtype(target_series)
        task = "clasificación" if (not is_numeric_target or n_unique <= 20) else "regresión"
        st.info(f"Se detectó tarea: {task} (unique={n_unique}, dtype={target_series.dtype})")

        features = [c for c in cols if c != target]
        sel_features = st.multiselect("Seleccionar features", features, default=features)

        test_size = st.slider("Proporción test", 0.1, 0.5, 0.25)
        n_estimators = st.slider("n_estimators (RandomForest)", 10, 500, 100)
        random_state = st.number_input("random_state", value=42, step=1)

        if st.button("Entrenar modelo"):
            with st.spinner("Entrenando..."):
                X = df[sel_features].copy()
                y = df[target].copy()

                numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

                num_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]) if numeric_cols else None

                cat_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
                ]) if categorical_cols else None

                transformers = []
                if numeric_cols:
                    transformers.append(("num", num_pipeline, numeric_cols))
                if categorical_cols:
                    transformers.append(("cat", cat_pipeline, categorical_cols))

                preprocessor = ColumnTransformer(transformers=transformers)

                ModelClass = RandomForestClassifier if task == "clasificación" else RandomForestRegressor
                model = Pipeline([
                    ("preprocessor", preprocessor),
                    ("estimator", ModelClass(n_estimators=n_estimators, random_state=int(random_state)))
                ])

                mask = y.notnull()
                X = X[mask]
                y = y[mask]

                for c in categorical_cols:
                    X[c] = X[c].astype(str)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if task == "clasificación":
                    acc = accuracy_score(y_test, y_pred)
                    st.success(f"Accuracy en test: {acc:.4f}")
                    st.write("Reporte de clasificación:")
                    st.text(classification_report(y_test, y_pred))

                    cm = confusion_matrix(y_test, y_pred)
                    st.write("Matriz de confusión:")
                    st.dataframe(pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test)))
                else:
                    r2 = r2_score(y_test, y_pred)
                    rmse = mean_squared_error(y_test, y_pred, squared=False)
                    st.success(f"R2: {r2:.4f} — RMSE: {rmse:.4f}")

                try:
                    feat_names = get_feature_names(model.named_steps['preprocessor'], numeric_cols, categorical_cols)
                except Exception:
                    feat_names = sel_features
                try:
                    importances = model.named_steps['estimator'].feature_importances_
                    fi = pd.DataFrame({"feature": feat_names, "importance": importances})
                    fi = fi.sort_values("importance", ascending=False).head(30)
                    st.write("Importancia de features (top):")
                    st.dataframe(fi.reset_index(drop=True))
                except Exception as e:
                    st.info("No fue posible extraer importancias de features: " + str(e))

                model_path = "modelo_entrenado.joblib"
                joblib.dump(model, model_path)
                with open(model_path, "rb") as f:
                    st.download_button("Descargar modelo entrenado (.joblib)", f, file_name=model_path, mime="application/octet-stream")
    else:
        st.info("Seleccione una columna objetivo para activar el panel de modelado.")

st.sidebar.markdown("---")
st.sidebar.markdown("Creado con ❤️ — Versión Colab-friendly")
