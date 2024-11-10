# analysis_descriptive.py

import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from data_loading import list_commands, load_waveforms

DATASET_PATH = '../../data/mini_speech_commands' 
data_dir = pathlib.Path(DATASET_PATH)

def analyze_audio_commands():
    commands = list_commands()
    audio_counts = {}

    # Contando a quantidade de áudios por comando
    for command in commands:
        command_path = os.path.join(data_dir, command)
        audio_files = [f for f in os.listdir(command_path) if f.endswith('.wav')]
        audio_counts[command] = len(audio_files)

    # Criar gráfico da quantidade de áudios por comando
    plt.figure(figsize=(10, 6))
    plt.bar(audio_counts.keys(), audio_counts.values(), color='skyblue')
    plt.title('Quantidade de Áudios por Comando')
    plt.xlabel('Comandos')
    plt.ylabel('Quantidade de Áudios')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Salvar a imagem na pasta /out/analysis_descriptive/
    os.makedirs('out/analysis_descriptive', exist_ok=True)
    plt.savefig('out/analysis_descriptive/quantidade_audios_por_comando.png')
    plt.close()  # Fecha a figura para liberar memória

    # Exibir a quantidade de áudios em cada comando
    print("Quantidade de áudios por comando:")
    for command, count in audio_counts.items():
        print(f"{command}: {count} áudios")

    return audio_counts

def plot_audio_samples(waveforms):
    commands = list_commands()
    fig, axs = plt.subplots(len(commands), 1, figsize=(10, 5 * len(commands)))

    # Usando a variável waveforms diretamente
    for i, command in enumerate(commands):
        # Obtendo a lista de formas de onda para o comando atual
        command_waveforms = waveforms.get(command, [])

        # Itera sobre cada forma de onda na lista
        for j, waveform in enumerate(command_waveforms):
            axs[i].plot(waveform, label=f'{command}_{j}')  # Nome fictício para cada áudio

        axs[i].set_title(f"Áudios do comando: {command}")
        axs[i].legend()

    plt.tight_layout()
    plt.savefig('out/analysis_descriptive/audio_samples.png')
    plt.show()
