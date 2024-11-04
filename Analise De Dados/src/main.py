# main.py

import data_loading
import audio_representation
import analysis_descriptive

print("\n \n \n \n")

# Carregar e explorar dados
data_dir = data_loading.download_and_extract_data()
commands = data_loading.list_commands()
waveforms = data_loading.load_waveforms() # Carregar formas de onda

# Durações dos áudios
data_loading.explore_durations(commands)

# Gerando a representação dos áudios
# audio_representation.generate_all_plots(waveforms)

# Gráficos de Análise Descritiva
analysis_descriptive.analyze_audio_commands()
analysis_descriptive.plot_audio_samples(waveforms)