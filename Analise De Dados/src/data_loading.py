# data_loading.py

import os
import pathlib
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
import zipfile

# Caminho do dataset e configuração de seeds para consistência nos resultados
DATASET_PATH = '../../data/mini_speech_commands' 
data_dir = pathlib.Path(DATASET_PATH)
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

def download_and_extract_data():
    """Baixa e extrai o dataset se ele ainda não estiver presente no diretório especificado."""
    
    # Cria o diretório 'data' se ele não existir
    if not data_dir.parent.exists():
        os.makedirs(data_dir.parent)
        print(f"Criado o diretório: {data_dir.parent}")

    if not data_dir.exists():
        print("Baixando o dataset...")
        file_path = tf.keras.utils.get_file(
            'mini_speech_commands.zip',
            origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
            cache_dir=str(data_dir.parent),
            cache_subdir='.')

        print("Download completo.")

        # Extraindo o arquivo zip manualmente
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir.parent)  # Extrai no diretório pai
            print(f"Extraído em: {data_dir.parent}")

        # Caminho da pasta extraída
        mini_speech_dir = data_dir.parent / 'mini_speech_commands'
        if mini_speech_dir.exists() and mini_speech_dir.is_dir():
            shutil.move(str(mini_speech_dir), str(data_dir))
            print(f"Movido: {mini_speech_dir} para {data_dir}")
        else:
            print(f"[ERRO] A pasta esperada não foi encontrada: {mini_speech_dir}")

        # Remover o arquivo zip após a extração
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removido o arquivo: {file_path}")

        # Remover pastas indesejadas
        unwanted_dirs = ['__MACOSX']
        for dir_name in unwanted_dirs:
            unwanted_path = data_dir.parent / dir_name
            if unwanted_path.exists() and unwanted_path.is_dir():
                shutil.rmtree(unwanted_path)
                print(f"Removido: {unwanted_path}")

    return data_dir


# Função para listar os comandos disponíveis no dataset
def list_commands():
    """Lista todos os comandos de áudio disponíveis no dataset."""
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    commands = commands[commands != 'README.md']
    print("Comandos disponíveis:", commands)
    return commands

# Função para explorar a distribuição das durações dos áudios e salvar um gráfico
def explore_durations(commands):
    """
    Analisa a distribuição das durações dos áudios para cada comando, 
    gera um gráfico da distribuição e salva no diretório 'out'.
    """
    durations = []  # Lista para armazenar a duração de cada áudio

    # Itera sobre os comandos e calcula a duração dos arquivos de áudio
    for command in commands:
        files = list(data_dir.glob(f'{command}/*.wav'))
        for file in files:
            if file.exists():
                # Decodifica o áudio e calcula a duração em segundos
                audio, _ = tf.audio.decode_wav(tf.io.read_file(str(file)))
                durations.append(len(audio) / 16000)  # Frequência de amostragem de 16000 Hz
            else:
                print(f"Arquivo não encontrado: {file}")
    
    # Criação do gráfico e diretório de saída
    if not os.path.exists('out'):
        os.makedirs('out')

    plt.figure(figsize=(10, 6))
    sns.histplot(durations, kde=True)
    plt.xlabel('Duração (s)')
    plt.ylabel('Frequência')
    plt.title('Distribuição das Durações dos Áudios')

    # Salvando o gráfico em um arquivo
    plt.savefig('out/data_loading/duracao_audios.png')
    plt.close()  # Fecha a figura para liberar memória
    print("Gráfico salvo em out/data_loading/duracao_audios.png")

# Função para carregar todos os arquivos de áudio do dataset e convertê-los em formas de onda
def load_waveforms():
    """
    Carrega os arquivos de áudio de cada comando do dataset e converte para formas de onda.
    
    Parâmetros:
        data_dir (pathlib.Path): O diretório raiz onde os comandos estão armazenados.
    
    Retorna:
        dict: Dicionário com cada comando como chave e lista de formas de onda como valor.
    """

    waveforms = {}  # Dicionário para armazenar as formas de onda de cada comando

    # Itera pelos diretórios de cada comando e carrega os arquivos de áudio
    commands = [d for d in data_dir.iterdir() if d.is_dir()]
    for command in commands:
        command_name = command.name
        waveforms[command_name] = []

        # Carrega e converte cada arquivo de áudio para forma de onda
        for audio_file in command.glob('*.wav'):
            audio_binary = tf.io.read_file(str(audio_file))
            waveform, _ = tf.audio.decode_wav(audio_binary)
            waveforms[command_name].append(waveform.numpy().flatten())  # Converte para array e adiciona à lista

    return waveforms
