# data_processing.py

import os
import librosa
import numpy as np

# Caminho para os arquivos de áudio
DATA_DIR = "../../../data/mini_speech_commands"
OUTPUT_DIR = "./data"  # Diretório onde os arquivos de saída serão salvos
FIXED_SHAPE = (128, 128)  # Tamanho fixo para os espectrogramas
MAX_SAMPLES_PER_LABEL = 1000  # Número máximo de amostras por comando

def extract_spectrogram(file_path, n_fft=2048, hop_length=512):
    """
    Extrai o espectrograma de um arquivo de áudio.
    
    Args:
        file_path (str): Caminho para o arquivo de áudio.
        n_fft (int): Número de pontos para a FFT.
        hop_length (int): Número de amostras entre cada frame de FFT.
        
    Returns:
        np.ndarray: Espectrograma em escala logarítmica.
    """
    # Carrega o arquivo de áudio
    y, sr = librosa.load(file_path, sr=None)
    # Gera o espectrograma de potência
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    # Converte para escala logarítmica (decibéis)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

def pad_or_truncate(spectrogram, shape=FIXED_SHAPE):
    """
    Ajusta o espectrograma para um tamanho fixo, preenchendo ou truncando conforme necessário.
    
    Args:
        spectrogram (np.ndarray): O espectrograma a ser ajustado.
        shape (tuple): A forma desejada (altura, largura).
        
    Returns:
        np.ndarray: Espectrograma ajustado.
    """
    padded_spectrogram = np.zeros(shape)
    padded_spectrogram[:min(shape[0], spectrogram.shape[0]), :min(shape[1], spectrogram.shape[1])] = \
        spectrogram[:shape[0], :shape[1]]
    return padded_spectrogram

def load_data(data_dir=DATA_DIR):
    """
    Carrega os dados de áudio e extrai espectrogramas com rótulos.
    
    Args:
        data_dir (str): Caminho para o diretório de dados.
        
    Returns:
        tuple: Arrays `X` (espectrogramas) e `y` (rótulos).
    """
    X, y = [], []
    
    # Itera sobre os diretórios de comandos
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            # Lista todos os arquivos de áudio no diretório do comando
            files = [file for file in os.listdir(label_dir) if file.endswith('.wav')]
            
            # Limita a quantidade de arquivos a 1000
            files = files[:MAX_SAMPLES_PER_LABEL]
            
            # Processa cada arquivo de áudio
            for file_name in files:
                file_path = os.path.join(label_dir, file_name)
                spectrogram = extract_spectrogram(file_path)
                # Ajusta o espectrograma para o tamanho fixo
                spectrogram_fixed = pad_or_truncate(spectrogram)
                # Achata o espectrograma em uma única dimensão
                X.append(spectrogram_fixed.flatten())
                y.append(label)
    
    # Converte listas para arrays numpy
    X = np.array(X)
    y = np.array(y)
    
    # Salva os arrays no diretório OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # Cria a pasta se não existir
    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
    
    print(f"Dados salvos em {OUTPUT_DIR}: 'X.npy' e 'y.npy'")
    return X, y
