import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd

# Carregar dados processados
conf_matrix = pd.read_csv("confusion_matrix.csv", index_col=0)
scatter_data = pd.read_csv("scatter_data.csv")

COMMANDS = conf_matrix.columns.tolist()

# Inicializar o app Dash
app = dash.Dash(__name__)

# Layout do dashboard
app.layout = html.Div(children=[
    html.H1(children='Dashboard de Classificação de Comandos de Voz'),

    html.Div(children='''
        Este dashboard exibe a matriz de confusão e o gráfico de dispersão das previsões do modelo.
    '''),

    dcc.Graph(
        id='confusion-matrix',
        figure=px.imshow(conf_matrix.values, 
                         labels=dict(x="Predicted", y="True", color="Count"),
                         x=COMMANDS, 
                         y=COMMANDS,
                         title="Matriz de Confusão")
    ),

    dcc.Graph(
        id='scatter-plot',
        figure=px.scatter(scatter_data, x='PCA1', y='PCA2', color='Predicted', symbol='True',
                          title="Gráfico de Dispersão das Previsões com PCA")
    )
])

# Executar o app
if __name__ == '__main__':
    app.run_server(debug=True)
