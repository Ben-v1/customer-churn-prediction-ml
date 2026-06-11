import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Churn Score App",
    page_icon="📉",
    layout="centered",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_xgboost_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "churn_model_metadata.json"

st.title("📉 Previsão de Churn de Clientes")
st.write(
    "Simulador para estimar a probabilidade de churn de um cliente. "
    "O modelo deve ser treinado no notebook antes de usar este app."
)

if not MODEL_PATH.exists():
    st.error(
        "Modelo não encontrado. Execute primeiro o notebook `notebooks/customer_churn_prediction.ipynb` "
        "para gerar `models/churn_xgboost_pipeline.joblib`."
    )
    st.stop()

model = joblib.load(MODEL_PATH)

threshold = 0.50
if METADATA_PATH.exists():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    threshold = float(metadata.get("best_threshold_validation_f1", 0.50))

st.sidebar.header("Dados do cliente")

age = st.sidebar.slider("Idade", 18, 80, 35)
gender = st.sidebar.selectbox("Gênero", ["Female", "Male"])
tenure = st.sidebar.slider("Tempo de relacionamento", 1, 60, 12)
usage_frequency = st.sidebar.slider("Frequência de uso", 1, 30, 10)
support_calls = st.sidebar.slider("Chamados de suporte", 0, 10, 2)
payment_delay = st.sidebar.slider("Atraso de pagamento", 0, 30, 5)
subscription_type = st.sidebar.selectbox("Tipo de assinatura", ["Basic", "Standard", "Premium"])
contract_length = st.sidebar.selectbox("Duração do contrato", ["Monthly", "Quarterly", "Annual"])
total_spend = st.sidebar.number_input("Gasto total", min_value=0.0, value=500.0, step=50.0)
last_interaction = st.sidebar.slider("Dias desde a última interação", 1, 30, 10)

input_df = pd.DataFrame([
    {
        "age": age,
        "gender": gender,
        "tenure": tenure,
        "usage_frequency": usage_frequency,
        "support_calls": support_calls,
        "payment_delay": payment_delay,
        "subscription_type": subscription_type,
        "contract_length": contract_length,
        "total_spend": total_spend,
        "last_interaction": last_interaction,
    }
])

# Mesma engenharia de atributos usada no notebook.
input_df["avg_spend_per_month"] = input_df["total_spend"] / (input_df["tenure"] + 1)
input_df["support_calls_per_month"] = input_df["support_calls"] / (input_df["tenure"] + 1)
input_df["payment_delay_rate"] = input_df["payment_delay"] / (input_df["tenure"] + 1)
input_df["engagement_score"] = input_df["usage_frequency"] / (input_df["last_interaction"] + 1)
input_df["high_support_calls"] = (input_df["support_calls"] >= 5).astype(int)
input_df["long_payment_delay"] = (input_df["payment_delay"] >= 15).astype(int)
input_df["low_usage"] = (input_df["usage_frequency"] <= 10).astype(int)
input_df["recently_inactive"] = (input_df["last_interaction"] >= 20).astype(int)
input_df["month_to_month_contract"] = input_df["contract_length"].str.lower().str.contains("month").astype(int)

probability = float(model.predict_proba(input_df)[:, 1][0])
prediction = int(probability >= threshold)

st.subheader("Resultado")
st.metric("Probabilidade estimada de churn", f"{probability:.2%}")
st.caption(f"Threshold usado: {threshold:.4f}")

if prediction == 1:
    st.warning("Cliente classificado como alto risco de churn.")
    st.write("Sugestão: priorizar contato proativo, revisar chamados de suporte e avaliar oferta de retenção.")
else:
    st.success("Cliente classificado como baixo risco de churn.")
    st.write("Sugestão: manter acompanhamento normal e monitorar sinais de mudança no engajamento.")

with st.expander("Ver dados enviados ao modelo"):
    st.dataframe(input_df)

st.markdown("---")
st.caption("© Brenno Gomes")
