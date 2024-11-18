# Voice Command Classification Projects

Este repositório contém três projetos distintos para a classificação de comandos de voz utilizando diferentes abordagens de aprendizado de máquina: Máquina de Vetores de Suporte (SVM), Árvores de Decisão e Redes Neurais Convolucionais (CNN). Cada projeto visa a construção e avaliação de um modelo para reconhecimento de comandos de voz a partir de espectrogramas de áudio.

## Sumário
- [Voice Command Classification with Support Vector Machine (SVM)](#voice-command-classification-with-support-vector-machine-svm)
- [Voice Command Classification with Decision Tree](#voice-command-classification-with-decision-tree)
- [Voice Command Classification Project with Convolutional Neural Network (CNN)](#voice-command-classification-project-with-convolutional-neural-network-cnn)
- [Como Executar os Projetos](#como-executar-os-projetos)
- [Requisitos](#requisitos)
- [Estrutura de Diretórios](#estrutura-de-diretórios)

## Voice Command Classification with Support Vector Machine (SVM)

Este projeto visa a construção e avaliação de um modelo de aprendizado de máquina para o reconhecimento de comandos de voz. O processo envolve o pré-processamento dos dados de áudio, treinamento de um modelo de Máquina de Vetores de Suporte (SVM) para classificação de espectrogramas de áudio e a avaliação do modelo utilizando um conjunto de dados de teste.

### Arquivos e Funções
- **`data_processing.py`**: Contém funções para pré-processar os arquivos de áudio, convertendo-os em espectrogramas log-mel.
- **`main.py`**: Executa o fluxo principal do projeto, incluindo o carregamento dos dados, treinamento do modelo SVM, avaliação do modelo e geração de gráficos.

### Resultados de Avaliação
- **Acurácia**: 53.56%
- **Relatório de Classificação**: Exibe precisão, recall e F1-score para cada classe de comando de voz.
- **Matriz de Confusão**: Visualiza o desempenho do modelo em cada classe.
- **Gráfico de Dispersão**: Visualiza a distribuição das previsões e ajuda a entender a separabilidade das classes.

## Voice Command Classification with Decision Tree

Este projeto visa a construção e avaliação de um modelo de aprendizado de máquina para reconhecimento de comandos de voz, utilizando Árvores de Decisão. O processo envolve o pré-processamento dos dados de áudio, treinamento do modelo e a avaliação do desempenho do modelo com base em um conjunto de dados de teste.

### Arquivos e Funções
- **`data_processing.py`**: Responsável pelo pré-processamento dos dados de áudio, convertendo-os em espectrogramas log-mel.
- **`main.py`**: Orquestra as etapas principais do projeto, incluindo o carregamento dos dados, treinamento do modelo de Árvore de Decisão, avaliação do modelo e geração dos gráficos de desempenho.

### Resultados de Avaliação
- **Acurácia**: 41.38%
- **Relatório de Classificação**: Exibe precisão, recall e F1-score para cada classe de comando de voz.
- **Matriz de Confusão**: Visualiza o desempenho do modelo em cada classe.
- **Gráfico de Dispersão**: Exibe a distribuição das previsões em um gráfico 2D utilizando PCA para reduzir a dimensionalidade.

## Voice Command Classification Project with Convolutional Neural Network (CNN)

Este projeto visa a construção e avaliação de um modelo de aprendizado de máquina para reconhecimento de comandos de voz. O processo envolve o pré-processamento dos dados de áudio, treinamento de um modelo convolucional para classificação de espectrogramas de áudio e a avaliação do modelo utilizando um conjunto de dados de teste.

### Arquivos e Funções
- **`model.py`**: Define a arquitetura do modelo de rede neural convolucional (CNN) para classificação dos comandos de voz.
- **`evaluation.py`**: Carrega o modelo treinado e avalia sua performance utilizando um conjunto de dados de teste, gerando um relatório de classificação e uma matriz de confusão.
- **`training.py`**: Carrega os dados, divide-os em conjuntos de treinamento e validação e treina o modelo CNN.
- **`data_preprocessing.py`**: Contém as funções para pré-processar os arquivos de áudio, convertendo-os em espectrogramas log-mel.
- **`main.py`**: Orquestra as principais etapas do projeto, permitindo executar o pré-processamento dos dados, criar o modelo CNN e treinar o modelo.

### Resultados de Avaliação
- **Acurácia**: 88.45%
- **Relatório de Classificação**: Exibe precisão, recall e F1-score para cada classe de comando de voz.
- **Matriz de Confusão**: Visualiza o desempenho do modelo em cada classe.
- **Gráfico de Dispersão**: Reduz a dimensionalidade dos dados de avaliação para permitir uma visualização mais clara das relações entre os comandos.

## Como Executar os Projetos

1. **Clone o repositório**:
    ```bash
    git clone https://github.com/FabioHenriqueFarias/Projeto-de-IA-e-AD.git
    cd Projeto-de-IA-e-AD
    ```

2. **Instale as dependências**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Execute o pré-processamento dos dados e treine o modelo**:
    - Para SVM:
        ```bash
        cd IA/maquinas-de-vetores-de-suporte/src
        python main.py
        ```
    - Para Árvores de Decisão:
        ```bash
        cd IA/arvores-de-decisoes/src
        python main.py
        ```
    - Para CNN:
        ```bash
        cd IA/redes-neurais-convolucionais/src
        python main.py
        ```

## Requisitos

Este projeto exige o Python 3.x e as seguintes dependências:
- numpy
- librosa
- tensorflow
- scikit-learn
- matplotlib
- seaborn

Instale as dependências com o comando:
```bash
pip install -r requirements.txt