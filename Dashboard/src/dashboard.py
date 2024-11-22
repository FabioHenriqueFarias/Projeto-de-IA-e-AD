# dashboard.py

from flask import Flask, render_template
import pandas as pd
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px

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

# Função para carregar imagem
def load_image(file_path):
    if file_path is None:
        return None
    return file_path if os.path.exists(file_path) else None

# Caminhos dos arquivos
ROOT_DIR = os.path.expanduser("~/Documentos/UNA/A3")
duration_data_path = os.path.join(ROOT_DIR, "Analise De Dados", "src", "out", "data_loading", "duracao_audios.csv")
confusion_matrix_path = os.path.join(ROOT_DIR, "IA", "redes-neurais-convolucionais", "src", "out", "confusion_matrix.png")
scatter_plot_path = os.path.join(ROOT_DIR, "IA", "redes-neurais-convolucionais", "src", "out", "scatter_plot.png")

# Carregar dados
duration_data = load_csv_data(duration_data_path)

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
    if os.path.exists(confusion_matrix_path):
        children.append(html.Img(src=confusion_matrix_path, style={'width': '50%'}))
    else:
        children.append(html.Div("Matriz de Confusão do CNN não encontrada."))
    
    if os.path.exists(scatter_plot_path):
        children.append(html.Img(src=scatter_plot_path, style={'width': '50%'}))
    else:
        children.append(html.Div("Gráfico de Dispersão do CNN não encontrado."))
    
    return children

if __name__ == '__main__':
    app.run(debug=True)