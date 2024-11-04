# audio_representation.py

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Função para plotar a forma de onda de um áudio
def plot_waveform(waveform, title="Forma de Onda"):
    """Plota a forma de onda de um áudio."""
    plt.plot(waveform)
    plt.title(title)  # Define o título do gráfico
    plt.xlabel("Amostras")  # Rótulo do eixo X
    plt.ylabel("Amplitude")  # Rótulo do eixo Y
    plt.grid()  # Adiciona uma grade ao gráfico
    plt.tight_layout()  # Melhora o layout do gráfico

# Função para gerar e plotar um espectrograma a partir da forma de onda
def plot_spectrogram(waveform, title="Espectrograma"):
    """Gera e plota um espectrograma."""
    # Calcula o espectrograma usando a Transformada de Fourier de Curto Prazo (STFT)
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)  # Obtém a magnitude do espectrograma
    plt.imshow(spectrogram.numpy(), aspect='auto', origin='lower', cmap='viridis')  # Plota o espectrograma
    plt.title(title)  # Define o título do gráfico
    plt.xlabel("Tempo")  # Rótulo do eixo X
    plt.ylabel("Frequência")  # Rótulo do eixo Y
    plt.colorbar()  # Adiciona uma barra de cor ao lado do gráfico

# Função para gerar e salvar gráficos de forma de onda e espectrograma para um áudio específico
def plot_waveform_and_spectrogram(waveform, command_name, audio_file_name):
    """
    Gera e salva gráficos de forma de onda e espectrograma para um áudio específico.

    Parâmetros:
        waveform (np.array): A forma de onda do áudio.
        command_name (str): O nome do comando para o áudio.
        audio_file_name (str): Nome do arquivo de áudio, usado para salvar os gráficos.
    """
    # Caminho de saída para os gráficos
    output_dir = f'out/audio_representation/{command_name}'  # Define o diretório para o comando específico
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Cria o diretório se não existir

    # Configurações do gráfico
    plt.figure(figsize=(12, 6))

    # Gráfico da forma de onda
    plt.subplot(2, 1, 1)  # Cria um subplot para a forma de onda
    plot_waveform(waveform, title=f"Forma de Onda - {command_name} - {audio_file_name}")

    # Gráfico do espectrograma
    plt.subplot(2, 1, 2)  # Cria um subplot para o espectrograma
    plot_spectrogram(waveform, title=f"Espectrograma - {command_name} - {audio_file_name}")

    # Salvando o gráfico
    plt.savefig(f'{output_dir}/{audio_file_name}.png')  # Salva o gráfico no diretório correspondente
    plt.close()  # Fecha o gráfico para liberar memória
    print(f"Gráfico salvo para {command_name}/{audio_file_name}.png")  # Mensagem de confirmação

# Função para gerar e salvar gráficos de exemplo para um arquivo de áudio
def generate_example_plots(file_path):
    """Gera e salva gráficos de exemplo para um arquivo de áudio."""
    
    # Verifica se o arquivo existe antes de processá-lo
    if not os.path.exists(file_path):
        print(f"Arquivo não encontrado: {file_path}")  # Mensagem de erro
        return  # Sai da função se o arquivo não existir

    audio, _ = tf.audio.decode_wav(tf.io.read_file(file_path))  # Lê e decodifica o arquivo de áudio
    audio = audio.numpy().flatten()  # Converte para array 1D

    plt.figure(figsize=(12, 6))  # Define o tamanho da figura

    # Plot da forma de onda
    plt.subplot(2, 1, 1)  # Cria um subplot para a forma de onda
    plot_waveform(audio)  # Plota a forma de onda
    
    # Geração do espectrograma
    plt.subplot(2, 1, 2)  # Cria um subplot para o espectrograma
    plot_spectrogram(audio)  # Plota o espectrograma

    # Salva o gráfico em um arquivo
    plt.savefig('out/audio_representation/example_plots.png')  # Salva o gráfico de exemplo
    plt.close()  # Fecha a figura para liberar memória
    print("Gráfico de exemplo salvo em out/audio_representation/example_plots.png")  # Mensagem de confirmação

# Função para gerar e salvar gráficos para todos os arquivos de áudio em todos os comandos
def generate_all_plots(waveforms):
    """
    Gera e salva gráficos para todos os arquivos de áudio em todos os comandos.

    Parâmetros:
        waveforms (dict): Dicionário com os nomes dos comandos como chaves e listas de formas de onda como valores.
    """
    for command_name, wave_list in waveforms.items():  # Itera sobre os comandos e suas formas de onda
        for i, waveform in enumerate(wave_list):  # Itera sobre as formas de onda
            audio_file_name = f"audio_{i + 1}"  # Define o nome do arquivo de áudio
            plot_waveform_and_spectrogram(waveform, command_name, audio_file_name)  # Gera e salva os gráficos

    print("Todos os gráficos foram gerados e salvos com sucesso.")  # Mensagem de confirmação final
