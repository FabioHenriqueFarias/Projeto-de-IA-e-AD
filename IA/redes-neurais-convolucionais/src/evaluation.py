# evaluation.py

import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import preprocess_audio

print("\n \n \n")

# Caminho para o modelo salvo e dados de avaliação
MODEL_PATH = "speech_command_model.h5"
EVAL_DATA_PATH = "data/assessment/"  # Diretório contendo todos os arquivos .wav de avaliação
COMMANDS = ["down", "go", "left", "no", "right", "stop", "up", "yes"]

# Carregar o modelo salvo
def load_trained_model(model_path=MODEL_PATH):
    model = load_model(model_path)
    print("Modelo carregado com sucesso!")
    return model

# Carregar e processar os dados de avaliação
def load_eval_data(eval_data_path=EVAL_DATA_PATH, target_size=(128, 128)):
    x_eval = []
    y_eval = []

    for file_name in os.listdir(eval_data_path):
        if file_name.endswith(".wav"):
            # Extrair o rótulo do comando a partir do nome do arquivo
            command = file_name.split("_")[0]  # Tenta identificar o comando a partir do nome do arquivo
            if command not in COMMANDS:
                print(f"Aviso: Comando '{command}' no arquivo '{file_name}' não é reconhecido.")
                continue
            
            label = COMMANDS.index(command)
            file_path = os.path.join(eval_data_path, file_name)
            
            # Pré-processa o áudio e redimensiona o espectrograma
            spectrogram = preprocess_audio(file_path)
            if spectrogram.shape != target_size:
                spectrogram = np.resize(spectrogram, target_size)
            
            x_eval.append(spectrogram)
            y_eval.append(label)
    
    # Verifique se há dados para avaliação
    if len(x_eval) == 0:
        print("Nenhum dado de avaliação foi carregado. Verifique o diretório e os arquivos de entrada.")
        return None, None

    # Converter para arrays numpy e ajustar dimensão para o modelo
    x_eval = np.array(x_eval)
    y_eval = np.array(y_eval)
    x_eval = np.expand_dims(x_eval, -1)  # Adicionar canal para o formato (128, 128, 1)

    return x_eval, y_eval

# Função para avaliar o modelo
def evaluate_model(model, x_eval, y_eval):
    if x_eval is None or y_eval is None:
        print("Erro: Dados de avaliação não carregados corretamente.")
        return
    
    # Fazer previsões
    y_pred = model.predict(x_eval)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Exibir resultados
    print("Relatório de Classificação:\n", classification_report(y_eval, y_pred_classes, target_names=COMMANDS))
    print("Matriz de Confusão:\n", confusion_matrix(y_eval, y_pred_classes))

# Executa o processo de avaliação
if __name__ == "__main__":
    model = load_trained_model()
    x_eval, y_eval = load_eval_data()
    evaluate_model(model, x_eval, y_eval)
