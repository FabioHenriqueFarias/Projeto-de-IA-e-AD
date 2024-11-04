# Projeto de Classificação de Comandos de Voz com TensorFlow

## Sumário

- [Introdução](#introdução)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Execução](#execução)
    - [Pré-requisitos](#pré-requisitos)
    - [Como Executar o Projeto](#como-executar-o-projeto)
    - [Criação de um Ambiente Virtual](#criação-de-um-ambiente-virtual)
    - [Uso do pipx](#uso-do-pipx)
- [Estrutura de Diretórios](#estrutura-de-diretórios)

## Introdução

Este projeto tem como objetivo desenvolver um modelo de aprendizado de máquina para a classificação de comandos de voz, utilizando o renomado dataset **Mini Speech Commands**. Através de uma abordagem comparativa, exploramos o impacto de duas representações de áudio — **formas de onda** e **espectrogramas** — no desempenho do modelo. A análise detalhada que será apresentada neste projeto não só contribui para uma compreensão mais profunda sobre as características dos dados, mas também permite avaliar como diferentes representações de áudio influenciam a precisão na classificação de comandos de voz. 

## Estrutura do Projeto

A estrutura do projeto é organizada em dois diretórios principais, cada um contendo um README específico que orienta sobre a execução e configuração:

- **[Analise De Dados](https://github.com/FabioHenriqueFarias/Projeto-de-IA-e-AD/tree/main/Analise%20De%20Dados)**: Este diretório abriga scripts dedicados à análise descritiva e inferencial dos dados. As análises realizadas aqui fornecem insights sobre a distribuição dos comandos de voz e exploram visualmente as características dos espectrogramas e formas de onda.

- **[IA](https://github.com/FabioHenriqueFarias/Projeto-de-IA-e-AD/tree/main/IA)**: Este diretório é focado no desenvolvimento do modelo de aprendizado de máquina. Aqui você encontrará o modelo de **Rede Neural Convolucional (CNN)**, scripts para pré-processamento dos dados, geração de espectrogramas e o treinamento necessário para a classificação dos comandos de voz.

## Execução

Para executar o projeto, siga as instruções abaixo para instalação e execução dos scripts.

### Pré-requisitos

Antes de começar, verifique se você tem o Python 3.7 ou superior instalado, bem como os seguintes pacotes:

- **TensorFlow**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **SciPy**

Para instalar todas as dependências necessárias, execute o seguinte comando na raiz do projeto:

```bash
pip install -r requirements.txt
```

### Criação de um Ambiente Virtual

Recomenda-se o uso de um ambiente virtual para gerenciar as dependências do projeto. Para criar um ambiente virtual, siga os passos abaixo:

1. Navegue até o diretório raiz do projeto.
2. Crie um ambiente virtual executando o comando:

    ```bash
    python -m venv env
    ```

3. Ative o ambiente virtual:
    - No Windows:

        ```bash
        .\env\Scripts\activate
        ```

    - No macOS/Linux:

        ```bash
        source env/bin/activate
        ```

4. Após ativar o ambiente, instale as dependências necessárias:

    ```bash
    pip install -r requirements.txt
    ```

### Uso do pipx

Se você deseja instalar ferramentas Python de linha de comando que não precisam de um ambiente virtual, o `pipx` é uma excelente opção. Para instalar e usar `pipx`, siga os passos abaixo:

1. Instale o `pipx` (se ainda não tiver):

    ```bash
    python -m pip install --user pipx
    python -m pipx ensurepath
    ```

2. Reinicie seu terminal ou execute `source ~/.bashrc` (ou equivalente) para que o `pipx` seja reconhecido.
3. Para instalar um pacote com `pipx`, como o TensorFlow (ou qualquer outro pacote que você precise para execução):

    ```bash
    pipx install <nome-do-pacote>
    ```


### Como Executar o Projeto

1. **Instalação das Dependências**: Assegure-se de que todos os pacotes necessários foram instalados conforme descrito acima.
2. **Seleção do Diretório**: Navegue até o diretório desejado, **[Analise De Dados](Analise De Dados)** ou **[IA](IA)**, dependendo do seu foco, seja na análise ou no treinamento do modelo.
3. **Execução Principal**: Para iniciar o fluxo completo do projeto, execute o arquivo `main.py`:

    ```bash
    python main.py
    ```

## Estrutura de Diretórios

```plaintext
Projeto-de-IA-e-AD/
│
├── Analise De Dados/              # Diretório dedicado à análise dos dados, contendo relatórios, gráficos e insights obtidos
│
├── IA/                            # Diretório que abriga os modelos de inteligência artificial, scripts de treinamento e inferência
│
└── requirements.txt               # Arquivo que lista as dependências necessárias para o projeto, facilitando a instalação e configuração do ambiente
```

---

Este projeto oferece uma abordagem abrangente para o processamento e classificação de áudio por meio de **Redes Neurais Convolucionais (CNNs)**. A análise e comparação entre as representações de áudio possibilitam uma avaliação clara do impacto dos espectrogramas em relação às formas de onda. Este trabalho não apenas fundamenta melhorias futuras na precisão do reconhecimento de comandos de voz, mas também serve como uma base sólida para a exploração de técnicas avançadas em **machine learning**.
