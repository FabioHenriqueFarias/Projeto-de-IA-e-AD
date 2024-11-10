# model.py

import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape=(128, 128, 1), num_classes=8):
    model = models.Sequential([
        layers.Input(shape=input_shape),  # Define explicitamente o input_shape aqui

        # Primeira camada convolucional
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Segunda camada convolucional
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Terceira camada convolucional
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Achatar a saída para a camada densa
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        
        # Camada de saída para classificação
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compilar o modelo
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
