# Análise e Treinamento do Modelo de Comandos de Voz

Este projeto visa a construção e avaliação de um modelo de aprendizado de máquina para reconhecimento de comandos de voz. O processo envolve a pré-processamento dos dados de áudio, treinamento de um modelo convolucional para classificação de espectrogramas de áudio e a avaliação do modelo utilizando um conjunto de dados de teste.

## Sumário
- [Descrição do Projeto](#descrição-do-projeto)
- [Arquivos e Funções](#arquivos-e-funções)
  - [model.py](#modelpy)
  - [evaluation.py](#evaluationpy)
  - [training.py](#trainingpy)
  - [data_preprocessing.py](#data_preprocessingpy)
- [Pré-processamento de Dados](#pré-processamento-de-dados)
- [Treinamento do Modelo](#treinamento-do-modelo)
- [Avaliação do Modelo](#avaliação-do-modelo)
- [Como Executar o Projeto](#como-executar-o-projeto)
- [Requisitos](#requisitos)

## Descrição do Projeto

Este projeto se destina a treinar e avaliar um modelo de rede neural convolucional (CNN) para a classificação de comandos de voz, usando espectrogramas extraídos de gravações de áudio. O objetivo principal é criar um modelo capaz de classificar comandos de voz simples, como "up", "down", "left", "right", "yes", "no", "go" e "stop". O modelo é treinado utilizando espectrogramas log-mel, que são representações visuais do áudio em termos de suas frequências e amplitudes ao longo do tempo.

## Arquivos e Funções

### [model.py](https://github.com/FabioHenriqueFarias/Projeto-de-IA-e-AD/blob/main/Reconhecimento%20De%20Comandos%20De%20Voz/src/model.py)

Este arquivo define a arquitetura do modelo de rede neural convolucional (CNN) para classificação dos comandos de voz. A função `create_model()` constrói e compila o modelo com três camadas convolucionais seguidas de camadas densas para a classificação final.

- **`create_model(input_shape=(128, 128, 1), num_classes=8)`**: Cria e compila um modelo CNN.
  - **Parâmetros:**
    - `input_shape`: Define a forma da entrada (normalmente espectrogramas de 128x128).
    - `num_classes`: Número de classes (comandos de voz).
  - **Retorno:** Um modelo compilado, pronto para treinamento.

### [evaluation.py](https://github.com/FabioHenriqueFarias/Projeto-de-IA-e-AD/blob/main/Reconhecimento%20De%20Comandos%20De%20Voz/src/evaluation.py)

Este arquivo é responsável por carregar o modelo treinado e avaliar sua performance utilizando um conjunto de dados de teste. Ele também gera um relatório de classificação e uma matriz de confusão para avaliar a precisão do modelo.

- **`load_trained_model(model_path="speech_command_model.h5")`**: Carrega o modelo treinado a partir do caminho especificado.
- **`load_eval_data(eval_data_path="data/assessment/")`**: Carrega os dados de avaliação, processando os arquivos de áudio em espectrogramas.
- **`evaluate_model(model, x_eval, y_eval)`**: Avalia o modelo carregado com os dados de avaliação e exibe o relatório de classificação e matriz de confusão.

### [training.py](https://github.com/FabioHenriqueFarias/Projeto-de-IA-e-AD/blob/main/Reconhecimento%20De%20Comandos%20De%20Voz/src/training.py)

Este arquivo é responsável por carregar os dados, dividir em conjuntos de treinamento e validação, e treinar o modelo CNN. Após o treinamento, o modelo é salvo para posterior avaliação.

- **`load_data(data_dir="data/processed")`**: Carrega os dados processados (espectrogramas) para treinamento.
- **`train_model(epochs=10, batch_size=32)`**: Treina o modelo utilizando os dados carregados.
  - **Parâmetros:**
    - `epochs`: Número de épocas de treinamento.
    - `batch_size`: Tamanho do lote de dados durante o treinamento.
  - **Retorno:** O modelo treinado.

### [data_preprocessing.py](https://github.com/FabioHenriqueFarias/Projeto-de-IA-e-AD/blob/main/Reconhecimento%20De%20Comandos%20De%20Voz/src/data_preprocessing.py)

Este arquivo contém as funções para pré-processar os arquivos de áudio, convertendo-os em espectrogramas log-mel, que são então usados para treinamento e avaliação do modelo.

- **`preprocess_audio(file_path, target_sr=16000)`**: Converte o arquivo de áudio em um espectrograma log-mel.
- **`save_spectrogram(spectrogram, save_path)`**: Salva o espectrograma processado no formato `.npy`.
- **`process_all_commands()`**: Processa todos os comandos de voz e salva seus espectrogramas correspondentes.

## Pré-processamento de Dados

O pré-processamento dos dados envolve a conversão dos arquivos de áudio em espectrogramas log-mel. Esses espectrogramas são então redimensionados para um formato adequado para a entrada do modelo.

- **Carregamento do áudio**: Usamos a biblioteca `librosa` para carregar os arquivos de áudio e normalizar para uma taxa de amostragem padrão.
- **Conversão para espectrograma**: O áudio é convertido para um espectrograma log-mel, que captura as características espectrais mais relevantes para a tarefa de classificação.
- **Salvamento dos espectrogramas**: Cada espectrograma é salvo em formato `.npy` para posterior uso no treinamento do modelo.

## Treinamento do Modelo

O treinamento do modelo é realizado utilizando uma rede neural convolucional (CNN), que é ideal para tarefas de classificação de imagens, como espectrogramas de áudio.

- **Divisão dos dados**: Os dados são divididos em conjuntos de treinamento e validação, com 80% dos dados usados para treinamento e 20% para validação.
- **Treinamento do modelo**: O modelo é treinado utilizando o algoritmo de otimização `Adam` e a função de perda `sparse_categorical_crossentropy`, com a métrica de precisão.
- **Salvamento do modelo**: Após o treinamento, o modelo é salvo no formato `.h5` para que possa ser carregado e avaliado posteriormente.

## Avaliação do Modelo

A avaliação do modelo é feita utilizando um conjunto de dados de avaliação, que é processado de forma similar ao conjunto de treinamento (convertido em espectrogramas). O modelo é carregado e usado para fazer previsões sobre os dados de avaliação. Os resultados incluem:

- **Relatório de classificação**: Exibe a precisão, recall e F1-score para cada classe de comando de voz.
- **Matriz de confusão**: Mostra o desempenho do modelo, visualizando os acertos e erros para cada classe de comando.

## Como Executar o Projeto

1. **Clone o repositório**:
    ```bash
    git clone https://github.com/FabioHenriqueFarias/Projeto-de-IA-e-AD.git
    cd Projeto-de-IA-e-AD
    ```
2. **Instale as dependências**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Execute o pré-processamento dos dados**:
    ```bash
    python src/data_preprocessing.py
    ```
    Isso irá processar os arquivos de áudio e salvar os espectrogramas.
4. **Treine o modelo**:
    ```bash
    python src/training.py
    ```
5. **Avalie o modelo**:
    ```bash
    python src/evaluation.py
    ```

## Requisitos

- Python 3.7 ou superior
- Bibliotecas: `tensorflow`, `librosa`, `numpy`, `matplotlib`, `scikit-learn`

## Estrutura de Diretórios

```plaintext
Projeto-de-IA-e-AD/
│
├── src/                            # Código fonte do projeto
│   ├── model.py                    # Arquitetura do modelo CNN
│   ├── evaluation.py               # Avaliação do modelo
│   ├── training.py                 # Treinamento do modelo
│   ├── data_preprocessing.py       # Pré-processamento dos dados de áudio
│
└── data/                           # Diretório contendo dados do projeto
    ├── processed/                  # Espectrogramas processados para treinamento
    └── assessment/                 # Dados de avaliação (arquivos .wav para avaliar o modelo)
│
├── requirements.txt               # Dependências do projeto
└── README.md                      # Documentação do projeto
```

Este projeto oferece um pipeline completo para o treinamento e avaliação de um modelo de aprendizado de máquina para o reconhecimento de comandos de voz a partir de espectrogramas de áudio.