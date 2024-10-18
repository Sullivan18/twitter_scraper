import subprocess
import shlex

def executar_scraping(username):
    # Define o comando para executar o scraping
    command = shlex.split(f"python scraper -t 10 -u {username}")

    try:
        # Executa o comando e aguarda a conclusão
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Scraping executado com sucesso.")
        print("Saída do comando:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar o comando: {e}")
        print("Saída do erro:", e.stderr)

# Exemplo de uso
executar_scraping("CazeTVOficial")
