import os
import pandas as pd
import matplotlib.pyplot as plt

def find_file(directory, filename):
    """Busca um arquivo específico em um diretório e subdiretórios."""
    for root, _, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None

def load_csv_data(file_path):
    """Carrega um arquivo CSV em um DataFrame do Pandas."""
    if file_path is None:
        return None
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return None

def load_image(file_path):
    """Carrega uma imagem de gráfico se o arquivo existir."""
    if file_path is None:
        return None
    return file_path if os.path.exists(file_path) else None

def list_directory(directory):
    """Lista todos os arquivos e diretórios em um diretório."""
    for root, dirs, files in os.walk(directory):
        level = root.replace(directory, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

import os

# Diretório raiz do projeto
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
print(f"ROOT_DIR calculado: {ROOT_DIR}")

# Diretórios e arquivos esperados
duration_data_path = find_file(os.path.join(ROOT_DIR, "Analise de Dados"), "duracao_audios.csv")
confusion_matrix_path = find_file(os.path.join(ROOT_DIR, "IA/redes-neurais-convolucionais/src/out"), "confusion_matrix.png")
scatter_plot_path = find_file(os.path.join(ROOT_DIR, "IA/redes-neurais-convolucionais/src/out"), "scatter_plot.png")
classification_report_path = find_file(os.path.join(ROOT_DIR, "IA/redes-neurais-convolucionais/src/out"), "classification_report.txt")

# Adicionando prints para depuração
print(f"Duration Data Path: {duration_data_path}")
print(f"Confusion Matrix Path: {confusion_matrix_path}")
print(f"Scatter Plot Path: {scatter_plot_path}")
print(f"Classification Report Path: {classification_report_path}")


# Carregar dados
duration_data = load_csv_data(duration_data_path)
confusion_matrix_image = load_image(confusion_matrix_path)
scatter_plot_image = load_image(scatter_plot_path)

# Atualização no dashboard
print("Dashboard de Análise de Dados e Modelos")
print("Uma visão centralizada dos resultados")

print("\n📊 Análise dos Dados")
if duration_data is not None:
    print("Distribuição das Durações dos Áudios")
    print(duration_data.describe())
else:
    print("Gráfico de distribuição das durações não encontrado.")

print("\n🧠 Resultados dos Modelos")
print("\nÁrvore de Decisão")
# Adicione dados da Árvore de Decisão quando disponíveis.

print("\nSVM")
# Adicione dados do SVM quando disponíveis.

print("\nCNN")
if confusion_matrix_image:
    print(f"Matriz de Confusão - CNN: {confusion_matrix_image}")
else:
    print("Matriz de Confusão do CNN não encontrada.")

if scatter_plot_image:
    print(f"Gráfico de Dispersão - CNN: {scatter_plot_image}")
else:
    print("Gráfico de Dispersão do CNN não encontrado.")