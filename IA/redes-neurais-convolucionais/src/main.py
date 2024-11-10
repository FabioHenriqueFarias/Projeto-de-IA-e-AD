# main.py

from data_preprocessing import process_all_commands
from model import create_model
from training import train_model

print("\n \n \n")

# Executar o pré-processamento
def preprocess_data():
    print("Iniciando o pré-processamento dos dados...")
    process_all_commands()
    print("Pré-processamento concluído.")

# Criar o modelo
def build_model():
    print("Criando o modelo CNN...")
    model = create_model()
    model.summary()

if __name__ == "__main__":
    preprocess_data() # só precisa ser executado uma vez
    # build_model() pode ser chamado para inspecionar o modelo
    # train_model()
