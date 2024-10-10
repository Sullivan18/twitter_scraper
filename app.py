from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os
import logging
import pandas as pd
import shlex
import re
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from unidecode import unidecode
import warnings

# Importar a função save_content_and_analyze_sentiment
from save_content_and_analyze_sentiment import save_content_and_analyze_sentiment

# Configuração do logger com nível DEBUG e inclusão de timestamp nos logs
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Variáveis de ambiente
TWEETS_FOLDER = os.getenv('TWEETS_FOLDER', './tweets/')  # Diretório dos arquivos CSV
SCRAPER_TIMEOUT = int(os.getenv('SCRAPER_TIMEOUT', 90))   # Timeout para o subprocesso

# Expressão regular para validar o username do Twitter
USERNAME_REGEX = re.compile(r'^[A-Za-z0-9_]{1,15}$')

app = FastAPI()

class UserRequest(BaseModel):
    username: str

def validate_username(username: str) -> bool:
    """Valida se o nome de usuário está no formato correto"""
    return bool(USERNAME_REGEX.match(username))

def sanitize_username(username: str) -> str:
    """Remove espaços em branco ou caracteres indesejados no nome de usuário."""
    return username.strip()

def execute_scraper(username: str) -> subprocess.CompletedProcess:
    """Executa o comando do scraper de forma segura usando shlex e com timeout."""
    command = shlex.split(f"python scraper -t 3 -u {username}")
    try:
        logger.info(f"Executing scraper command: {' '.join(command)}")
        # Adicionando timeout ao subprocesso
        result = subprocess.run(command, capture_output=True, text=True, timeout=SCRAPER_TIMEOUT)
        return result
    except subprocess.TimeoutExpired:
        logger.error("O comando do scraper demorou muito tempo para ser executado.")
        raise HTTPException(status_code=500, detail="O scraper demorou muito para responder.")
    except FileNotFoundError:
        logger.error("O comando do scraper não foi encontrado.")
        raise HTTPException(status_code=500, detail="Script do scraper não encontrado.")
    except Exception as e:
        logger.error(f"Erro ao executar o scraper: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao executar o scraper: {str(e)}")

def validate_csv_file(filepath: str) -> pd.DataFrame:
    """Valida se o arquivo CSV não está vazio ou corrompido."""
    try:
        # Tenta carregar o CSV
        df = pd.read_csv(filepath)
        if df.empty:
            raise HTTPException(status_code=500, detail="O arquivo CSV está vazio.")
        return df
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=500, detail="O arquivo CSV está vazio ou corrompido.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar o CSV: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "API is working. Use the /scrape endpoint to start scraping."}

@app.post("/scrape")
def scrape(user_request: UserRequest):
    start_time = time.time()
    username = sanitize_username(user_request.username)

    # Verifica se o username é válido
    if not validate_username(username):
        raise HTTPException(status_code=400, detail="Invalid Twitter username. Must be 1-15 characters, containing letters, numbers, or underscores.")

    try:
        logger.info(f"Starting scraper for user: {username}")
        
        # Executa o scraper
        result = execute_scraper(username)
        logger.info(f"Scraper output: {result.stdout}")
        logger.error(f"Scraper errors: {result.stderr}")

        # Localizar o último CSV gerado na pasta de tweets
        csv_files = sorted([f for f in os.listdir(TWEETS_FOLDER) if f.endswith('.csv')], key=lambda x: os.path.getmtime(os.path.join(TWEETS_FOLDER, x)))
        if not csv_files:
            raise HTTPException(status_code=500, detail="No CSV files found.")
        
        latest_csv = os.path.join(TWEETS_FOLDER, csv_files[-1])

        # Validação do CSV
        df = validate_csv_file(latest_csv)

        # Chama a função save_content_and_analyze_sentiment
        results = save_content_and_analyze_sentiment(latest_csv)
        
        # Log do tempo de execução
        elapsed_time = time.time() - start_time
        logger.info(f"Scraping for user {username} completed in {elapsed_time:.2f} seconds.")
        
        return {"message": f"Scraping for user {username} completed.", "results": results}
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(status_code=400, detail="Username is required.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
