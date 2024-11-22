from flask import Flask, render_template
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import os

# Inicializar o app Flask
app = Flask(__name__)

# Configuração do Dash
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dashboard/')
dash_app.layout = html.Div([
    html.H1("Dashboard de Análise de Dados e Modelos"),
    dcc.Tabs([
        dcc.Tab(label='Análise dos Dados', children=[
            html.Div(id='data-analysis-content')
        ]),
        dcc.Tab(label='Resultados dos Modelos', children=[
            html.Div(id='model-results-content')
        ])
    ])
])

# Função para carregar dados
def load_csv_data(file_path):
    if file_path is None:
        return None
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        return None

# Caminhos dos arquivos
ROOT_DIR = os.path.abspath(".")
duration_data_path = os.path.join(ROOT_DIR, "data", "duracao_audios.csv")
confusion_matrix_path = os.path.join(ROOT_DIR, "confusion_matrix.csv")
scatter_plot_path = os.path.join(ROOT_DIR, "scatter_data.csv")

# Carregar dados
duration_data = load_csv_data(duration_data_path)
conf_matrix = load_csv_data(confusion_matrix_path)
scatter_data = load_csv_data(scatter_plot_path)

# Rota principal
@app.route('/')
def index():
    return render_template('index.html')

# Callback para atualizar o conteúdo da aba de análise de dados
@dash_app.callback(
    dash.dependencies.Output('data-analysis-content', 'children'),
    []
)
def update_data_analysis():
    if duration_data is not None:
        fig = px.histogram(duration_data, x='duration', title='Distribuição das Durações dos Áudios')
        return dcc.Graph(figure=fig)
    else:
        return html.Div("Gráfico de distribuição das durações não encontrado.")

# Callback para atualizar o conteúdo da aba de resultados dos modelos
@dash_app.callback(
    dash.dependencies.Output('model-results-content', 'children'),
    []
)
def update_model_results():
    children = []
    if conf_matrix is not None:
        fig_conf_matrix = px.imshow(conf_matrix.values, 
                                    labels=dict(x="Predicted", y="True", color="Count"),
                                    x=conf_matrix.columns.tolist(), 
                                    y=conf_matrix.index.tolist(),
                                    title="Matriz de Confusão")
        children.append(dcc.Graph(figure=fig_conf_matrix))
    else:
        children.append(html.Div("Matriz de Confusão do CNN não encontrada."))
    
    if scatter_data is not None:
        fig_scatter = px.scatter(scatter_data, x='PCA1', y='PCA2', color='Predicted', symbol='True',
                                 title="Gráfico de Dispersão das Previsões com PCA")
        children.append(dcc.Graph(figure=fig_scatter))
    else:
        children.append(html.Div("Gráfico de Dispersão do CNN não encontrado."))
    
    return children

if __name__ == '__main__':
    app.run(debug=True)