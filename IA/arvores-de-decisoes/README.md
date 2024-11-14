Aqui está o README.md com base na sua descrição do projeto usando Árvores de Decisão, seguindo a estrutura solicitada:

---

# Voice Command Classification with Decision Tree

Este projeto visa a construção e avaliação de um modelo de aprendizado de máquina para reconhecimento de comandos de voz, utilizando Árvores de Decisão. O processo envolve o pré-processamento dos dados de áudio, treinamento do modelo e a avaliação do desempenho do modelo com base em um conjunto de dados de teste.

## Sumário
- [Descrição do Projeto](#descrição-do-projeto)
- [Arquivos e Funções](#arquivos-e-funções)
  - [data_processing.py](#data_processingpy)
  - [main.py](#mainpy)
- [Pré-processamento de Dados](#pré-processamento-de-dados)
- [Treinamento do Modelo](#treinamento-do-modelo)
- [Avaliação do Modelo](#avaliação-do-modelo)
- [Como Executar o Projeto](#como-executar-o-projeto)
- [Resultados de Avaliação](#resultados-de-avaliação)
  - [Acurácia](#acurácia)
  - [Relatório de Classificação](#relatório-de-classificação)
  - [Matriz de Confusão](#matriz-de-confusão)
  - [Gráfico de Dispersão](#gráfico-de-dispersão)
- [Requisitos](#requisitos)

## Descrição do Projeto

Este projeto tem como objetivo treinar e avaliar um modelo de Árvores de Decisão para a classificação de comandos de voz extraídos de arquivos de áudio. O modelo utiliza espectrogramas log-mel extraídos das gravações de áudio, que são representações visuais das frequências presentes no áudio ao longo do tempo. O modelo de árvore de decisão classifica os comandos de voz em categorias como "up", "down", "left", "right", "yes", "no", "go" e "stop".

## Arquivos e Funções

### data_processing.py

Este script é responsável pelo pré-processamento dos dados de áudio, convertendo-os em espectrogramas log-mel, que são utilizados como entrada para o modelo de aprendizado de máquina.

- **`extract_spectrogram(file_path, n_fft=2048, hop_length=512)`**: Extrai o espectrograma de um arquivo de áudio.
- **`pad_or_truncate(spectrogram, shape=FIXED_SHAPE)`**: Ajusta o espectrograma para um tamanho fixo, preenchendo ou truncando conforme necessário.
- **`load_data(data_dir=DATA_DIR)`**: Carrega os dados de áudio, extrai os espectrogramas e rótulos, e os salva em arrays numpy para uso posterior.

### main.py

Este script orquestra as etapas principais do projeto, incluindo o carregamento dos dados, treinamento do modelo de Árvore de Decisão, avaliação do modelo e geração dos gráficos de desempenho.

- **Carregamento dos dados**: Utiliza o método `load_data()` de `data_processing.py` para carregar os espectrogramas e os rótulos dos comandos.
- **Treinamento do modelo**: Usa a classe `DecisionTreeClassifier` do `scikit-learn` para treinar o modelo.
- **Avaliação do modelo**: Avalia a performance do modelo utilizando a acurácia, o relatório de classificação, a matriz de confusão e o gráfico de dispersão.
- **Visualização**: Gera e salva a matriz de confusão e o gráfico de dispersão, representando a distribuição das previsões.

## Pré-processamento de Dados

O pré-processamento consiste em carregar os arquivos de áudio, extrair espectrogramas log-mel e ajustar o tamanho dos espectrogramas para um formato fixo. Esse processo é realizado nas funções presentes no arquivo `data_processing.py`.

## Treinamento do Modelo

O treinamento é feito utilizando o modelo de Árvore de Decisão da biblioteca `scikit-learn`. O modelo é treinado com os dados de treinamento, e sua performance é avaliada utilizando os dados de teste. A acurácia e outros parâmetros de desempenho são calculados.

## Avaliação do Modelo

A avaliação do modelo é feita utilizando um conjunto de dados de teste. Os resultados incluem:

- **Acurácia**: Mede a proporção de previsões corretas do modelo.
- **Relatório de Classificação**: Exibe as métricas de precisão, recall e F1-score para cada classe de comando de voz.
- **Matriz de Confusão**: Visualiza o desempenho do modelo em cada classe de comando.
- **Gráfico de Dispersão**: Exibe a distribuição das previsões em um gráfico 2D utilizando PCA para reduzir a dimensionalidade.

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
    python main.py
    ```
    Isso irá processar os arquivos de áudio, extrair os espectrogramas e salvar os dados no diretório `./data`.

4. **Treine e avalie o modelo**:
    O script `main.py` também treina e avalia o modelo automaticamente, exibindo a acurácia e gerando os gráficos de avaliação.

## Resultados de Avaliação

Após a execução do script `main.py`, obtivemos os seguintes resultados:

### Acurácia
- **41.38%**

### Relatório de Classificação

| Comando | Precision | Recall | F1-Score | Suporte |
| ------- | --------- | ------ | -------- | ------- |
| down    | 0.42      | 0.42   | 0.42     | 207     |
| go      | 0.35      | 0.29   | 0.32     | 222     |
| left    | 0.29      | 0.36   | 0.32     | 183     |
| no      | 0.29      | 0.34   | 0.31     | 196     |
| right   | 0.49      | 0.45   | 0.47     | 203     |
| stop    | 0.60      | 0.55   | 0.57     | 200     |
| up      | 0.36      | 0.39   | 0.37     | 197     |
| yes     | 0.59      | 0.54   | 0.57     | 192     |

- **Média Macro:** Precisão: 0.42 | Recall: 0.42 | F1-score: 0.42
- **Média Ponderada:** Precisão: 0.42 | Recall: 0.41 | F1-score: 0.42

### Matriz de Confusão

A matriz de confusão é gerada automaticamente e salva como uma imagem. Ela visualiza o desempenho do modelo, mostrando onde ele acerta ou confunde os comandos. As células na diagonal principal indicam boas predições, enquanto as células fora da diagonal mostram onde o modelo comete erros.

![Matriz de Confusão](./out/confusion_matrix.png)

### Gráfico de Dispersão

O gráfico de dispersão é gerado após reduzir a dimensionalidade dos dados de teste para 2D utilizando PCA. Cada ponto no gráfico representa uma previsão, com a cor indicando a classe prevista.

![Gráfico de Dispersão](./out/decision_tree_scatter_plot.png)

## Requisitos

Este projeto requer o Python 3.x e as seguintes dependências:

- numpy
- librosa
- scikit-learn
- matplotlib

Instale as dependências com o comando:

```bash
pip install -r requirements.txt
```

---

Esse README estrutura e detalha as etapas de seu projeto com base nas árvores de decisão. Ele explica as funções principais de cada script, como rodar o código e os resultados obtidos com a avaliação do modelo.