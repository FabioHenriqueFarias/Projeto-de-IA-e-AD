# dashboard.py
from flask import Flask, render_template
import pandas as pd
import os
import dash
from dash import dcc, html
import plotly.express as px

# Inicializar Flask
app = Flask(__name__)

# Configuração do Dash
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dashboard/')
dash_app.layout = html.Div([
    html.H1("Dashboard de Análise de Dados e Modelos"),
    dcc.Tabs([
        dcc.Tab(label='Análise dos Dados', children=[
            html.Div(id='data-analysis-content')
        ]),
        dcc.Tab(label='Resultados CNN', children=[
            html.Div(id='cnn-results-content')
        ])
    ])
])

# Função para carregar dados CSV
def load_csv_data(file_path):
    if file_path and os.path.exists(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Erro ao carregar arquivo {file_path}: {e}")
            return None
    else:
        print(f"Arquivo não encontrado: {file_path}")
        return None

# Caminhos dos arquivos
ROOT_DIR = os.getcwd()
paths = {
    "Matriz de Confusão": os.path.join(ROOT_DIR, "confusion_matrix.csv"),
    "Gráfico de Dispersão": os.path.join(ROOT_DIR, "scatter_data.csv"),
}

# Carregar dados
conf_matrix = load_csv_data(paths["Matriz de Confusão"])
scatter_data = load_csv_data(paths["Gráfico de Dispersão"])

# Rota principal do Flask
@app.route('/')
def index():
    return render_template('index.html')

# Callback para a aba de análise de dados
@dash_app.callback(
    dash.dependencies.Output('data-analysis-content', 'children'),
    []
)
def update_data_analysis():
    if scatter_data is not None:
        fig = px.scatter(scatter_data, x='PCA1', y='PCA2', color='Predicted', symbol='True',
                         title="Gráfico de Dispersão das Previsões com PCA")
        return dcc.Graph(figure=fig)
    else:
        return html.Div("Gráfico de dispersão não encontrado.")

# Callback para a aba de resultados CNN
@dash_app.callback(
    dash.dependencies.Output('cnn-results-content', 'children'),
    []
)
def update_cnn_results():
    if conf_matrix is not None:
        fig = px.imshow(conf_matrix.values, 
                        labels=dict(x="Predicted", y="True", color="Count"),
                        x=conf_matrix.columns.tolist(), 
                        y=conf_matrix.index.tolist(),
                        title="Matriz de Confusão")
        return dcc.Graph(figure=fig)
    else:
        return html.Div("Matriz de Confusão não encontrada.")

if __name__ == '__main__':
    app.run(debug=True)
