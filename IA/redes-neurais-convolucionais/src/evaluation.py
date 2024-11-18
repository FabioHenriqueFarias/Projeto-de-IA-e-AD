# evaluation.py

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from data_preprocessing import preprocess_audio

print("\n \n \n")

# Caminho para o modelo salvo e dados de avaliação
MODEL_PATH = "speech_command_model.h5"
EVAL_DATA_PATH = "../../../data/mini_speech_commands"  # Diretório contendo todos os arquivos .wav de avaliação
OUTPUT_DIR = "./out"
COMMANDS = ["down", "go", "left", "no", "right", "stop", "up", "yes"]

# Carregar o modelo salvo
def load_trained_model(model_path=MODEL_PATH):
    model = load_model(model_path)
    print("Modelo carregado com sucesso!")
    return model

# Carregar os dados de avaliação
def load_eval_data(eval_data_path=EVAL_DATA_PATH, target_size=(128, 128)):
    x_eval = []
    y_eval = []

    # Itera pelas pastas de comandos dentro do diretório de dados de avaliação
    for command in os.listdir(eval_data_path):
        command_path = os.path.join(eval_data_path, command)
        
        if not os.path.isdir(command_path):
            continue
        if command not in COMMANDS:
            print(f"Aviso: Comando '{command}' encontrado em '{command_path}' não é reconhecido.")
            continue
        
        label = COMMANDS.index(command)
        for file_name in os.listdir(command_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(command_path, file_name)
                spectrogram = preprocess_audio(file_path)
                if spectrogram.shape != target_size:
                    spectrogram = np.resize(spectrogram, target_size)
                x_eval.append(spectrogram)
                y_eval.append(label)

    if len(x_eval) == 0:
        print("Nenhum dado de avaliação foi carregado. Verifique o diretório e os arquivos de entrada.")
        return None, None

    x_eval = np.array(x_eval)
    y_eval = np.array(y_eval)
    x_eval = np.expand_dims(x_eval, -1)  # Adicionar a dimensão do canal (128, 128, 1)

    return x_eval, y_eval

# Função para salvar a matriz de confusão
def save_confusion_matrix(y_true, y_pred, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    conf_matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=COMMANDS)
    disp.plot(cmap='viridis', ax=ax, values_format='d')
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    print(f"Matriz de confusão salva em {output_dir}/confusion_matrix.png")

# Função para salvar o relatório de classificação
def save_classification_report(y_true, y_pred, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=COMMANDS)
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Relatório de classificação salvo em {report_path}")


# Função para salvar o gráfico de dispersão
def save_scatter_plot(x_eval, y_pred, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    pca = PCA(n_components=2)
    x_eval_pca = pca.fit_transform(x_eval.reshape(len(x_eval), -1))
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x_eval_pca[:, 0], x_eval_pca[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label="Classes")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Scatter Plot of Predictions with PCA-reduced Evaluation Data")
    plt.savefig(os.path.join(output_dir, "scatter_plot.png"))
    plt.close()
    print(f"Scatter plot salvo em {output_dir}/scatter_plot.png")

# Função para avaliar o modelo
def evaluate_model(model, x_eval, y_eval):
    if x_eval is None or y_eval is None:
        print("Erro: Dados de avaliação não carregados corretamente.")
        return
    
    y_pred = model.predict(x_eval)
    y_pred_classes = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_eval, y_pred_classes)
    print(f"Acurácia: {accuracy * 100:.2f}%")
    print("Relatório de Classificação:\n", classification_report(y_eval, y_pred_classes, target_names=COMMANDS))

    # Salvar a matriz de confusão
    save_confusion_matrix(y_eval, y_pred_classes)
    # Salvar o gráfico de dispersão
    save_scatter_plot(x_eval, y_pred_classes)
    # Salvar o relatório de classificação
    save_classification_report(y_eval, y_pred_classes)


# Executa o processo de avaliação
if __name__ == "__main__":
    model = load_trained_model()
    x_eval, y_eval = load_eval_data()
    evaluate_model(model, x_eval, y_eval)
