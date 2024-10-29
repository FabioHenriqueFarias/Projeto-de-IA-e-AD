# analysis.py

import matplotlib.pyplot as plt
from scipy import stats

def plot_model_performance(history):
    """Plota a performance do modelo durante o treinamento."""
    plt.plot(history.history['accuracy'], label='Acurácia de Treino')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.show()

def compare_accuracies(accuracy_waveform, accuracy_spectrogram):
    """Realiza um teste t para comparar as acurácias dos modelos."""
    t_stat, p_value = stats.ttest_ind(accuracy_waveform, accuracy_spectrogram)
    print("Estatística T:", t_stat)
    print("P-valor:", p_value)
    if p_value < 0.05:
        print("A diferença é estatisticamente significativa.")
    else:
        print("A diferença não é estatisticamente significativa.")


