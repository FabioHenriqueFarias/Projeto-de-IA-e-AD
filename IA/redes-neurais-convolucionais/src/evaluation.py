# evaluation.py

import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data_preprocessing import preprocess_audio

print("\n \n \n")

# Caminho para o modelo salvo e dados de avaliação
MODEL_PATH = "speech_command_model.h5"
EVAL_DATA_PATH = "../../../data/mini_speech_commands"  # Diretório contendo todos os arquivos .wav de avaliação
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

        # Verifica se o item é uma pasta, não um arquivo
        if not os.path.isdir(command_path):
            continue
        
        # Verifica se o comando está na lista de comandos conhecidos
        if command not in COMMANDS:
            print(f"Aviso: Comando '{command}' encontrado em '{command_path}' não é reconhecido.")
            continue
        
        # A partir do nome da pasta (que corresponde ao comando), pegamos o rótulo
        label = COMMANDS.index(command)

        # Agora vamos carregar os arquivos .wav dentro dessa pasta
        for file_name in os.listdir(command_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(command_path, file_name)
                
                # Processa o áudio e converte em espectrograma
                spectrogram = preprocess_audio(file_path)
                # Verifica e ajusta o tamanho do espectrograma
                if spectrogram.shape != target_size:
                    spectrogram = np.resize(spectrogram, target_size)

                # Adiciona o espectrograma e o rótulo na lista
                x_eval.append(spectrogram)
                y_eval.append(label)

    # Verifique se algum dado foi carregado
    if len(x_eval) == 0:
        print("Nenhum dado de avaliação foi carregado. Verifique o diretório e os arquivos de entrada.")
        return None, None

    # Converter para arrays numpy e ajustar a dimensão para o modelo
    x_eval = np.array(x_eval)
    y_eval = np.array(y_eval)
    x_eval = np.expand_dims(x_eval, -1)  # Adicionar a dimensão do canal (128, 128, 1)

    return x_eval, y_eval


# Função para avaliar o modelo
def evaluate_model(model, x_eval, y_eval):
    if x_eval is None or y_eval is None:
        print("Erro: Dados de avaliação não carregados corretamente.")
        return
    
    # Fazer previsões
    y_pred = model.predict(x_eval)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calcular a acurácia
    accuracy = accuracy_score(y_eval, y_pred_classes)
    print(f"Acurácia: {accuracy * 100:.2f}%")

    # Exibir resultados
    print("Relatório de Classificação:\n", classification_report(y_eval, y_pred_classes, target_names=COMMANDS))
    print("Matriz de Confusão:\n", confusion_matrix(y_eval, y_pred_classes))

# Executa o processo de avaliação
if __name__ == "__main__":
    model = load_trained_model()
    x_eval, y_eval = load_eval_data()
    evaluate_model(model, x_eval, y_eval)
