# training.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from model import create_model

# Diretório onde os espectrogramas estão salvos
DATA_DIR = "data/processed"

# Dicionário para mapear os comandos para rótulos numéricos
COMMANDS = ["down", "go", "left", "no", "right", "stop", "up", "yes"]

# Função para carregar dados
def load_data(data_dir=DATA_DIR, target_size=(128, 128)):
    x_data = []
    y_labels = []

    for command in os.listdir(data_dir):
        command_dir = os.path.join(data_dir, command)
        if not os.path.isdir(command_dir):
            continue
        # Aqui usamos o índice do comando na lista COMMANDS para gerar o rótulo
        label = COMMANDS.index(command)  # Encontra o índice do comando na lista COMMANDS
        for file in os.listdir(command_dir):
            file_path = os.path.join(command_dir, file)
            spectrogram = np.load(file_path)  # Carrega o espectrograma em numpy

            # Verificar o tamanho e redimensionar os espectrogramas
            if spectrogram.shape != target_size:
                spectrogram = np.resize(spectrogram, target_size)  # Redimensiona o espectrograma para (128, 128)

            x_data.append(spectrogram)
            y_labels.append(label)

    x_data = np.array(x_data)
    y_labels = np.array(y_labels)

    return x_data, y_labels

# Função para treinar o modelo
def train_model(epochs=10, batch_size=32):
    print("Carregando dados para treinamento...")
    x_data, y_labels = load_data()
    
    # Dividir dados em treino e validação
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_labels, test_size=0.2, random_state=42)

    # Expandir dimensões para incluir o canal (transformar para formato (128, 128, 1))
    x_train = np.expand_dims(x_train, -1)  # Adiciona o canal para a entrada do modelo
    x_val = np.expand_dims(x_val, -1)

    # Criar e treinar o modelo
    print("Treinando o modelo CNN...")
    model = create_model(input_shape=(128, 128, 1), num_classes=8)
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )

    # Salvar o modelo treinado
    model_path = "speech_command_model.h5"
    model.save(model_path)
    print(f"Modelo salvo como '{model_path}'")

    return model
