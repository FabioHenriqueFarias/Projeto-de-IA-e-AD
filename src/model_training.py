# model_training.py

import tensorflow as tf
from sklearn.model_selection import train_test_split

def preprocess_audio(audio):
    """Transforma o áudio em espectrograma para entrada no modelo."""
    spectrogram = tf.signal.stft(audio, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    return spectrogram

def create_model(input_shape):
    """Cria uma rede neural convolucional para classificação de comandos de voz."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(commands), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=10):
    """Treina o modelo usando dados de treino e validação."""
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))
    return history

# Dividir os dados em treino/validação
# (Exemplo de divisão; você terá que adaptar para seus dados)
# X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=SEED)
