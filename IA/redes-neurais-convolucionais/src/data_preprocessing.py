# data_preprocessing.py

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Definindo os caminhos para os dados
RAW_DATA_PATH = "../../../data/mini_speech_commands"
PROCESSED_DATA_PATH = "data/processed"
COMMANDS = ["down", "go", "left", "no", "right", "stop", "up", "yes"]

# Função para converter áudio em espectrograma
def preprocess_audio(file_path, target_sr=16000):
    # Carrega o áudio e converte para uma taxa de amostragem padrão
    audio, sr = librosa.load(file_path, sr=target_sr)
    
    # Converte o áudio para um espectrograma log-mel
    # Alterando a forma de passar os parâmetros para o melspectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=target_sr, n_mels=128)
    log_spectrogram = librosa.power_to_db(spectrogram)
    
    return log_spectrogram


# Função para salvar o espectrograma como arquivo .npy
def save_spectrogram(spectrogram, save_path):
    np.save(save_path, spectrogram)

# Processar todos os arquivos de áudio para cada comando
def process_all_commands():
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)
    
    for command in COMMANDS:
        command_path = os.path.join(RAW_DATA_PATH, command)
        save_dir = os.path.join(PROCESSED_DATA_PATH, command)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for file_name in os.listdir(command_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(command_path, file_name)
                
                # Processar o áudio e salvar o espectrograma
                spectrogram = preprocess_audio(file_path)
                save_path = os.path.join(save_dir, file_name.replace(".wav", ".npy"))
                save_spectrogram(spectrogram, save_path)
                
                print(f"Espectrograma salvo em {save_path}")