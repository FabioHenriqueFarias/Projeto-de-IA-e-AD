# process.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import confusion_matrix
import os
import sys

# Adicionar caminho de módulos externos
module_path = os.path.abspath('../../IA/redes-neurais-convolucionais/src')
if module_path not in sys.path:
    sys.path.append(module_path)

from evaluation import load_eval_data, COMMANDS

# Caminhos importantes
MODEL_PATH = os.path.abspath('../../IA/redes-neurais-convolucionais/src/speech_command_model.h5')
EVAL_DATA_PATH = os.path.abspath('../../data/mini_speech_commands')

# Carregar modelo
print("Carregando modelo...")
model = load_model(MODEL_PATH)
print("Modelo carregado com sucesso.")

# Carregar dados de avaliação
print("Carregando dados de avaliação...")
x_eval, y_eval = load_eval_data(EVAL_DATA_PATH)
print("Dados carregados com sucesso.")

# Fazer previsões
print("Fazendo previsões...")
predictions = model.predict(x_eval)
y_pred_classes = np.argmax(predictions, axis=1)

# Gerar matriz de confusão
print("Gerando matriz de confusão...")
conf_matrix = confusion_matrix(y_eval, y_pred_classes)

# Salvar matriz de confusão
pd.DataFrame(conf_matrix, index=COMMANDS, columns=COMMANDS).to_csv("confusion_matrix.csv", index=True)
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
scatter_data.to_csv("scatter_data.csv", index=False)
print("Dados PCA salvos.")
