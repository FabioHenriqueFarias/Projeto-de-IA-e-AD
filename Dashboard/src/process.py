# process.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import confusion_matrix
import os
import sys
import librosa

# Adicionar caminho de módulos externos
module_path = os.path.abspath('../IA/redes-neurais-convolucionais/src')
if module_path not in sys.path:
    sys.path.append(module_path)

# Caminhos importantes
MODEL_PATH = os.path.abspath('../../IA/redes-neurais-convolucionais/src/speech_command_model.h5')
EVAL_DATA_PATH = os.path.abspath('../../data/mini_speech_commands')

# Carregar modelo
print("Carregando modelo...")
model = load_model(MODEL_PATH)
print("Modelo carregado com sucesso.")

# Exibir a arquitetura do modelo
model.summary()

# Função para carregar dados de avaliação
def load_eval_data(eval_data_path, max_len=126):
    x_eval = []
    y_eval = []
    commands = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']
    for command in commands:
        command_path = os.path.join(eval_data_path, command)
        for file_name in os.listdir(command_path):
            file_path = os.path.join(command_path, file_name)
            audio, sr = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            # Truncar ou preencher os MFCCs para ter um comprimento fixo
            if mfcc.shape[1] > max_len:
                mfcc = mfcc[:, :max_len]
            else:
                mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
            mfcc = np.pad(mfcc, ((0, 126 - mfcc.shape[0]), (0, 0)), mode='constant')
            x_eval.append(mfcc)
            y_eval.append(commands.index(command))
    x_eval = np.array(x_eval)
    y_eval = np.array(y_eval)
    x_eval = x_eval[..., np.newaxis]
    
    print(f"Dados carregados: {x_eval.shape[0]} amostras")
    return x_eval, y_eval

# Carregar dados de avaliação
print("Carregando dados de avaliação...")
x_eval, y_eval = load_eval_data(EVAL_DATA_PATH, max_len=126)
print("Dados carregados com sucesso.")

# Fazer previsões
print("Fazendo previsões...")
predictions = model.predict(x_eval)
y_pred_classes = np.argmax(predictions, axis=1)

# Gerar matriz de confusão
print("Gerando matriz de confusão...")
conf_matrix = confusion_matrix(y_eval, y_pred_classes)

# Verificar se a matriz de confusão está correta
print("Matriz de confusão gerada:")
print(conf_matrix)

# Criar diretório para salvar os arquivos, se não existir
DATA_DIR = os.path.join(os.getcwd(), 'data')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Diretório '{DATA_DIR}' criado.")

# Salvar matriz de confusão
COMMANDS = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']
conf_matrix_df = pd.DataFrame(conf_matrix, index=COMMANDS, columns=COMMANDS)
conf_matrix_df.to_csv(os.path.join(DATA_DIR, "confusion_matrix.csv"), index=True)
print("Matriz de confusão salva.")

# PCA incremental
print("Aplicando PCA...")
ipca = IncrementalPCA(n_components=2)
x_eval_reshaped = x_eval.reshape(len(x_eval), -1)
x_eval_pca = ipca.fit_transform(x_eval_reshaped)

# Salvar dados PCA
scatter_data = pd.DataFrame({
    'PCA1': x_eval_pca[:, 0],
    'PCA2': x_eval_pca[:, 1],
    'Predicted': [COMMANDS[i] for i in y_pred_classes],
    'True': [COMMANDS[i] for i in y_eval]
})

# Verificar se os dados PCA estão corretos
print("Dados PCA preparados:")
print(scatter_data.head())

scatter_data.to_csv(os.path.join(DATA_DIR, "scatter_data.csv"), index=False)
print("Dados PCA salvos.")

# Função para rodar o processo completo
def run_evaluation():
    print("Iniciando o processo de avaliação...")
    # Carregar dados e fazer previsões
    load_eval_data(EVAL_DATA_PATH)
    print("Avaliação finalizada.")
