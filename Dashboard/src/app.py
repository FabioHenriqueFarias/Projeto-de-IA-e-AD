# app.py

from flask import Flask, render_template
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import numpy as np
import os

# Gerar dados de exemplo para scatter_data.csv
def generate_scatter_data(file_path):
    np.random.seed(0)
    data = {
        'PCA1': np.random.randn(100),
        'PCA2': np.random.randn(100),
        'Predicted': np.random.choice(['Classe1', 'Classe2'], 100),
        'True': np.random.choice(['Classe1', 'Classe2'], 100)
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Arquivo {file_path} gerado com sucesso.")

# Gerar dados de exemplo para matrizes de confusão
def generate_confusion_matrix(file_path, classes):
    np.random.seed(0)
    data = np.random.randint(0, 100, size=(len(classes), len(classes)))
    df = pd.DataFrame(data, index=classes, columns=classes)
    df.to_csv(file_path, index=True)
    print(f"Arquivo {file_path} gerado com sucesso.")

# Caminhos dos arquivos
ROOT_DIR = os.getcwd()
paths = {
    "Matriz de Confusão SVM": os.path.join(ROOT_DIR, "data", "confusion_matrix_svm.csv"),
    "Matriz de Confusão Árvore de Decisão": os.path.join(ROOT_DIR, "data", "confusion_matrix_decision_tree.csv"),
    "Matriz de Confusão Rede Neural Convolucional": os.path.join(ROOT_DIR, "data", "confusion_matrix_cnn.csv"),
    "Gráfico de Dispersão": os.path.join(ROOT_DIR, "data", "scatter_data.csv"),
}

# Classes de exemplo
classes = ['Classe1', 'Classe2', 'Classe3', 'Classe4']

# Gerar e salvar os dados
generate_scatter_data(paths["Gráfico de Dispersão"])
generate_confusion_matrix(paths["Matriz de Confusão SVM"], classes)
generate_confusion_matrix(paths["Matriz de Confusão Árvore de Decisão"], classes)
generate_confusion_matrix(paths["Matriz de Confusão Rede Neural Convolucional"], classes)

# Inicializar o app Flask
app = Flask(__name__)

# Configuração do Dash
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dashboard/')
dash_app.layout = html.Div([
    html.H1("Dashboard de Análise de Dados e Modelos"),
    dcc.Tabs([
        dcc.Tab(label='Análise dos Dados', children=[
            html.Div(id='data-analysis-content', className='tab-content')
        ]),
        dcc.Tab(label='Resultados SVM', children=[
            html.Div(id='svm-results-content', className='tab-content')
        ]),
        dcc.Tab(label='Resultados Árvore de Decisão', children=[
            html.Div(id='decision-tree-results-content', className='tab-content')
        ]),
        dcc.Tab(label='Resultados Rede Neural Convolucional', children=[
            html.Div(id='cnn-results-content', className='tab-content')
        ])
    ]),
    dcc.Interval(
        id='interval-component',
        interval=60*1000,  # Atualiza a cada 1 minuto
        n_intervals=0
    )
], style={'font-family': 'Roboto, sans-serif', 'background': '#f5f7fa', 'padding': '20px'})

# Função para carregar dados CSV
def load_csv_data(file_path):
    if os.path.exists(file_path):
        try:
            print(f"Carregando arquivo: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Erro ao carregar arquivo {file_path}: {e}")
            return None
    else:
        print(f"Arquivo não encontrado: {file_path}")
        return None

# Carregar dados
conf_matrix_svm = load_csv_data(paths["Matriz de Confusão SVM"])
conf_matrix_decision_tree = load_csv_data(paths["Matriz de Confusão Árvore de Decisão"])
conf_matrix_cnn = load_csv_data(paths["Matriz de Confusão Rede Neural Convolucional"])
scatter_data = load_csv_data(paths["Gráfico de Dispersão"])

# Verificar se os dados foram carregados corretamente
if conf_matrix_svm is not None:
    print(f"Matriz de Confusão SVM carregada com {conf_matrix_svm.shape[0]} linhas e {conf_matrix_svm.shape[1]} colunas.")
    conf_matrix_svm = conf_matrix_svm.apply(pd.to_numeric, errors='coerce').fillna(0)
else:
    print("Erro ao carregar a Matriz de Confusão SVM.")

if conf_matrix_decision_tree is not None:
    print(f"Matriz de Confusão Árvore de Decisão carregada com {conf_matrix_decision_tree.shape[0]} linhas e {conf_matrix_decision_tree.shape[1]} colunas.")
    conf_matrix_decision_tree = conf_matrix_decision_tree.apply(pd.to_numeric, errors='coerce').fillna(0)
else:
    print("Erro ao carregar a Matriz de Confusão Árvore de Decisão.")

if conf_matrix_cnn is not None:
    print(f"Matriz de Confusão Rede Neural Convolucional carregada com {conf_matrix_cnn.shape[0]} linhas e {conf_matrix_cnn.shape[1]} colunas.")
    conf_matrix_cnn = conf_matrix_cnn.apply(pd.to_numeric, errors='coerce').fillna(0)
else:
    print("Erro ao carregar a Matriz de Confusão Rede Neural Convolucional.")

if scatter_data is not None:
    print(f"Dados PCA carregados com {scatter_data.shape[0]} linhas e {scatter_data.shape[1]} colunas.")
else:
    print("Erro ao carregar os Dados PCA.")

# Rota principal do Flask
@app.route('/')
def index():
    return render_template('index.html')

# Callback para a aba de análise de dados
@dash_app.callback(
    dash.dependencies.Output('data-analysis-content', 'children'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_data_analysis(n_intervals):
    scatter_data = load_csv_data(paths["Gráfico de Dispersão"])
    if scatter_data is not None:
        scatter_fig = px.scatter(scatter_data, x='PCA1', y='PCA2', color='Predicted', symbol='True',
                                 title="Gráfico de Dispersão das Previsões com PCA",
                                 labels={'PCA1': 'Primeiro Componente Principal', 'PCA2': 'Segundo Componente Principal', 'Predicted': 'Classe Prevista', 'True': 'Classe Verdadeira'},
                                 template='plotly_white')
        scatter_fig.update_layout(title_font_size=20, title_x=0.5, margin=dict(l=20, r=20, t=40, b=20))

        hist_fig = px.histogram(scatter_data, x='Predicted', color='True',
                                title="Histograma das Classes Previstas",
                                labels={'Predicted': 'Classe Prevista', 'True': 'Classe Verdadeira'},
                                template='plotly_white')
        hist_fig.update_layout(title_font_size=20, title_x=0.5, margin=dict(l=20, r=20, t=40, b=20))

        return html.Div([
            dcc.Graph(figure=scatter_fig, className='dash-graph'),
            dcc.Graph(figure=hist_fig, className='dash-graph')
        ])
    else:
        return html.Div("Gráficos de análise de dados não encontrados.", className='dash-graph')

# Callback para a aba de resultados SVM
@dash_app.callback(
    dash.dependencies.Output('svm-results-content', 'children'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_svm_results(n_intervals):
    conf_matrix_svm = load_csv_data(paths["Matriz de Confusão SVM"])
    if conf_matrix_svm is not None:
        conf_matrix_svm = conf_matrix_svm.apply(pd.to_numeric, errors='coerce').fillna(0)
        conf_matrix_fig = px.imshow(conf_matrix_svm.values, 
                                    labels=dict(x="Classe Prevista", y="Classe Verdadeira", color="Contagem"),
                                    x=conf_matrix_svm.columns.tolist(), 
                                    y=conf_matrix_svm.index.tolist(),
                                    title="Matriz de Confusão SVM",
                                    color_continuous_scale='Blues',
                                    template='plotly_white')
        conf_matrix_fig.update_layout(title_font_size=20, title_x=0.5, margin=dict(l=20, r=20, t=40, b=20))

        bar_fig = px.bar(conf_matrix_svm.sum(axis=1).reset_index(), x='index', y=0,
                         title="Contagem de Classes Verdadeiras SVM",
                         labels={'index': 'Classe Verdadeira', 0: 'Contagem'},
                         template='plotly_white')
        bar_fig.update_layout(title_font_size=20, title_x=0.5, margin=dict(l=20, r=20, t=40, b=20))

        return html.Div([
            dcc.Graph(figure=conf_matrix_fig, className='dash-graph'),
            dcc.Graph(figure=bar_fig, className='dash-graph')
        ])
    else:
        return html.Div("Gráficos de resultados SVM não encontrados.", className='dash-graph')

# Callback para a aba de resultados Árvore de Decisão
@dash_app.callback(
    dash.dependencies.Output('decision-tree-results-content', 'children'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_decision_tree_results(n_intervals):
    conf_matrix_decision_tree = load_csv_data(paths["Matriz de Confusão Árvore de Decisão"])
    if conf_matrix_decision_tree is not None:
        conf_matrix_decision_tree = conf_matrix_decision_tree.apply(pd.to_numeric, errors='coerce').fillna(0)
        conf_matrix_fig = px.imshow(conf_matrix_decision_tree.values, 
                                    labels=dict(x="Classe Prevista", y="Classe Verdadeira", color="Contagem"),
                                    x=conf_matrix_decision_tree.columns.tolist(), 
                                    y=conf_matrix_decision_tree.index.tolist(),
                                    title="Matriz de Confusão Árvore de Decisão",
                                    color_continuous_scale='Blues',
                                    template='plotly_white')
        conf_matrix_fig.update_layout(title_font_size=20, title_x=0.5, margin=dict(l=20, r=20, t=40, b=20))

        bar_fig = px.bar(conf_matrix_decision_tree.sum(axis=1).reset_index(), x='index', y=0,
                         title="Contagem de Classes Verdadeiras Árvore de Decisão",
                         labels={'index': 'Classe Verdadeira', 0: 'Contagem'},
                         template='plotly_white')
        bar_fig.update_layout(title_font_size=20, title_x=0.5, margin=dict(l=20, r=20, t=40, b=20))

        return html.Div([
            dcc.Graph(figure=conf_matrix_fig, className='dash-graph'),
            dcc.Graph(figure=bar_fig, className='dash-graph')
        ])
    else:
        return html.Div("Gráficos de resultados Árvore de Decisão não encontrados.", className='dash-graph')

# Callback para a aba de resultados Rede Neural Convolucional
@dash_app.callback(
    dash.dependencies.Output('cnn-results-content', 'children'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_cnn_results(n_intervals):
    conf_matrix_cnn = load_csv_data(paths["Matriz de Confusão Rede Neural Convolucional"])
    if conf_matrix_cnn is not None:
        conf_matrix_cnn = conf_matrix_cnn.apply(pd.to_numeric, errors='coerce').fillna(0)
        conf_matrix_fig = px.imshow(conf_matrix_cnn.values, 
                                    labels=dict(x="Classe Prevista", y="Classe Verdadeira", color="Contagem"),
                                    x=conf_matrix_cnn.columns.tolist(), 
                                    y=conf_matrix_cnn.index.tolist(),
                                    title="Matriz de Confusão Rede Neural Convolucional",
                                    color_continuous_scale='Blues',
                                    template='plotly_white')
        conf_matrix_fig.update_layout(title_font_size=20, title_x=0.5, margin=dict(l=20, r=20, t=40, b=20))

        bar_fig = px.bar(conf_matrix_cnn.sum(axis=1).reset_index(), x='index', y=0,
                         title="Contagem de Classes Verdadeiras Rede Neural Convolucional",
                         labels={'index': 'Classe Verdadeira', 0: 'Contagem'},
                         template='plotly_white')
        bar_fig.update_layout(title_font_size=20, title_x=0.5, margin=dict(l=20, r=20, t=40, b=20))

        return html.Div([
            dcc.Graph(figure=conf_matrix_fig, className='dash-graph'),
            dcc.Graph(figure=bar_fig, className='dash-graph')
        ])
    else:
        return html.Div("Gráficos de resultados Rede Neural Convolucional não encontrados.", className='dash-graph')

if __name__ == '__main__':
    app.run(debug=True)