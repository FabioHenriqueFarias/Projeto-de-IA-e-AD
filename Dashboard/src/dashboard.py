import streamlit as st
import os
from PIL import Image

# Configuração do Streamlit
st.set_page_config(page_title="Dashboard - Análise e Modelos", layout="wide")

# Diretórios de saída
OUTPUT_DIR = "./out"

# Seções do Dashboard
st.title("Dashboard de Análise de Dados e Modelos")
st.markdown("### Uma visão centralizada dos resultados")

# 1. Análise dos Dados
st.header("📊 Análise dos Dados")
st.subheader("Distribuição das Durações dos Áudios")
durations_img_path = os.path.join(OUTPUT_DIR, "data_loading", "duracao_audios.png")

if os.path.exists(durations_img_path):
    st.image(durations_img_path, caption="Distribuição das Durações dos Áudios")
else:
    st.warning("Gráfico de distribuição das durações não encontrado.")

# 2. Modelos
st.header("🧠 Resultados dos Modelos")
model_tabs = st.tabs(["Árvore de Decisão", "SVM", "CNN"])

# Função para carregar métricas e gráficos
def display_model_results(model_name, conf_matrix_file, scatter_plot_file, metrics_text):
    st.subheader(f"Matriz de Confusão - {model_name}")
    if os.path.exists(conf_matrix_file):
        st.image(conf_matrix_file)
    else:
        st.warning(f"Matriz de Confusão do {model_name} não encontrada.")

    st.subheader(f"Gráfico de Dispersão - {model_name}")
    if os.path.exists(scatter_plot_file):
        st.image(scatter_plot_file)
    else:
        st.warning(f"Gráfico de Dispersão do {model_name} não encontrado.")
    
    st.subheader("Relatório de Classificação")
    st.text(metrics_text)

# Árvore de Decisão
with model_tabs[0]:
    display_model_results(
        "Árvore de Decisão",
        os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
        os.path.join(OUTPUT_DIR, "decision_tree_scatter_plot.png"),
        "Relatório gerado no treinamento da Árvore de Decisão."
    )

# SVM
with model_tabs[1]:
    display_model_results(
        "SVM",
        os.path.join(OUTPUT_DIR, "svm_confusion_matrix.png"),
        os.path.join(OUTPUT_DIR, "svm_scatter_plot.png"),
        "Relatório gerado no treinamento do SVM."
    )

# CNN
with model_tabs[2]:
    display_model_results(
        "CNN",
        os.path.join(OUTPUT_DIR, "confusion_matrix_cnn.png"),
        os.path.join(OUTPUT_DIR, "cnn_scatter_plot.png"),
        "Relatório gerado na avaliação do modelo CNN."
    )
