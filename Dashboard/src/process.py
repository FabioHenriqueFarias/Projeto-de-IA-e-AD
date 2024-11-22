import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import confusion_matrix
import os
import sys
import librosa
from tensorflow.keras import layers, models

# Adicionar caminho de módulos externos
module_path = os.path.abspath('../IA/redes-neurais-convolucionais/src')
if module_path not in sys.path:
    sys.path.append(module_path)

# Caminhos importantes
MODEL_PATH = os.path.abspath('../../IA/redes-neurais-convolucionais/src/speech_command_model.h5')
EVAL_DATA_PATH = os.path.abspath('../../data/mini_speech_commands')

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
            x_eval.append(mfcc)
            y_eval.append(commands.index(command))
    x_eval = np.array(x_eval)
    y_eval = np.array(y_eval)
    # Adicionar uma dimensão de canal
    x_eval = x_eval[..., np.newaxis]
    return x_eval, y_eval

# Carregar modelo
print("Carregando modelo...")
model = load_model(MODEL_PATH)
print("Modelo carregado com sucesso.")

# Exibir a arquitetura do modelo
model.summary()

# Carregar dados de avaliação
print("Carregando dados de avaliação...")
x_eval, y_eval = load_eval_data(EVAL_DATA_PATH, max_len=126)
print("Dados carregados com sucesso.")

# Garantir que a entrada tem a forma (samples, height, width, channels)
print(f"Forma de x_eval: {x_eval.shape}")

# Fazer previsões
print("Fazendo previsões...")
predictions = model.predict(x_eval)
y_pred_classes = np.argmax(predictions, axis=1)

# Gerar matriz de confusão
print("Gerando matriz de confusão...")
conf_matrix = confusion_matrix(y_eval, y_pred_classes)

# Salvar matriz de confusão
COMMANDS = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']
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


# -------------------
# ** Arquitetura do Modelo Ajustada **
# -------------------

# Model Definition: Ajuste de camadas para evitar problemas com redução excessiva das dimensões

def build_model(input_shape=(126, 126, 1), num_classes=8):
    model = models.Sequential([
        # Primeira camada convolucional
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Segunda camada convolucional
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Terceira camada convolucional
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Achatar a entrada para a camada densa
        layers.Flatten(),

        # Camada densa (fully connected)
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Construir e resumir o modelo
model = build_model()
model.summary()

