# main.py

from data_preprocessing import process_all_commands

# Executar o pré-processamento
def preprocess_data():
    print("Iniciando o pré-processamento dos dados...")
    process_all_commands()
    print("Pré-processamento concluído.")

if __name__ == "__main__":
    preprocess_data()