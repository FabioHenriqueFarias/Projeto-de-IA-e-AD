# main.py

import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # Substitui o SVM
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from data_processing import load_data

# Carrega os dados
X, y = load_data()

# Codifica os rótulos em valores numéricos
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializa e treina o modelo de Árvore de Decisão
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Avalia o modelo
y_pred = model.predict(X_test)

# Cálculo da acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy * 100:.2f}%")

# Relatório de Classificação
print("Relatório de Classificação:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Função para salvar a matriz de confusão
def save_confusion_matrix(y_true, y_pred, output_dir="./out"):
    os.makedirs(output_dir, exist_ok=True)
    conf_matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
    disp.plot(cmap='viridis', ax=ax, values_format='d')
    plt.title("Confusion Matrix of Decision Tree Predictions")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    print(f"Matriz de confusão salva em {output_dir}/confusion_matrix.png")

# Função para salvar o gráfico de dispersão
def save_scatter_plot(X_test, y_pred, output_dir="./out"):
    os.makedirs(output_dir, exist_ok=True)
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label="Classes")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Scatter Plot of Decision Tree Predictions with PCA-reduced Test Data")
    plt.savefig(os.path.join(output_dir, "decision_tree_scatter_plot.png"))
    plt.close()
    print(f"Scatter plot salvo em {output_dir}/decision_tree_scatter_plot.png")

# Salva a matriz de confusão e o gráfico de dispersão
save_confusion_matrix(y_test, y_pred)
save_scatter_plot(X_test, y_pred)
