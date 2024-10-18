from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings
import subprocess
import os
import logging
import pandas as pd
import shlex
import re
import time
import asyncio
from functools import lru_cache  # Para cache das requisições
from unidecode import unidecode
from dotenv import load_dotenv
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from requests.exceptions import Timeout
import requests  # Usado para a chamada à API Hugging Face
from save_content_and_analyze_sentiment import save_content_and_analyze_sentiment

# Carregar variáveis de ambiente
load_dotenv()

# Configuração das variáveis de ambiente
class Settings(BaseSettings):
    twitter_username: str = Field(..., env='TWITTER_USERNAME')
    twitter_password: str = Field(..., env='TWITTER_PASSWORD')
    tweets_folder: str = Field('./tweets/', env='TWEETS_FOLDER')
    scraper_timeout: int = Field(90, env='SCRAPER_TIMEOUT')
    host: str = Field('0.0.0.0', env='HOST')
    port: int = Field(8001, env='PORT')
    workers: int = Field(1, env='WORKERS')
    environment: str = Field('development', env='ENVIRONMENT')

    class Config:
        env_file = '.env'

# Instanciar as configurações
try:
    settings = Settings()
except ValidationError as e:
    print("Erro ao carregar variáveis de ambiente:")
    print(e.json())
    exit(1)

# Configuração básica do logger (apenas logs essenciais)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuração do middleware CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Limite para tarefas assíncronas simultâneas
semaphore = asyncio.Semaphore(5)

# Expressão regular para validar o username do Twitter
USERNAME_REGEX = re.compile(r'^[A-Za-z0-9_]{1,15}$')

class UserRequest(BaseModel):
    username: str

def validate_username(username: str) -> bool:
    """Valida o nome de usuário."""
    return bool(USERNAME_REGEX.match(username))

def sanitize_username(username: str) -> str:
    """Remove espaços em branco do nome de usuário."""
    return username.strip()

@lru_cache(maxsize=1000)  # Cache para até 1000 resultados
def query_api(text: str) -> dict:
    """Chama a API da Hugging Face com timeout e cache."""
    payload = {"inputs": text}
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/cardiffnlp/twitter-xlm-roberta-base-sentiment",
            headers={"Authorization": f"Bearer {settings.huggingface_token}"},
            json=payload,
            timeout=10  # Timeout de 10 segundos
        )
        response.raise_for_status()
        return response.json()
    except Timeout:
        logger.error("A requisição para a API da Hugging Face expirou.")
        raise HTTPException(status_code=504, detail="A requisição para a API expirou.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Erro na requisição para a API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro na API: {str(e)}")

async def limited_task(func, *args):
    """Executa uma tarefa com limite de concorrência."""
    async with semaphore:
        return await asyncio.to_thread(func, *args)

def execute_scraper(username: str) -> subprocess.CompletedProcess:
    """Executa o scraper com timeout."""
    command = shlex.split(f"python scraper -t 10 -u {username}")
    try:
        logger.info(f"Executing scraper command: {' '.join(command)}")
        return subprocess.run(command, capture_output=True, text=True, timeout=settings.scraper_timeout)
    except subprocess.TimeoutExpired:
        logger.error("O scraper demorou muito tempo para responder.")
        raise HTTPException(status_code=500, detail="O scraper demorou muito para responder.")
    except FileNotFoundError:
        logger.error("Script do scraper não encontrado.")
        raise HTTPException(status_code=500, detail="Script do scraper não encontrado.")
    except Exception as e:
        logger.error(f"Erro ao executar o scraper: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao executar o scraper: {str(e)}")

def validate_csv_file(filepath: str) -> pd.DataFrame:
    """Valida e processa o CSV."""
    try:
        chunks = pd.read_csv(filepath, chunksize=1000)
        return pd.concat(chunk for chunk in chunks if not chunk.empty)
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=500, detail="O arquivo CSV está vazio.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar o CSV: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "API is working. Use the /scrape endpoint to start scraping."}

@app.post("/scrape")
async def scrape(user_request: UserRequest):
    start_time = time.time()
    username = sanitize_username(user_request.username)

    if not validate_username(username):
        raise HTTPException(status_code=400, detail="Invalid Twitter username. Must be 1-15 characters, containing letters, numbers, or underscores.")

    try:
        logger.info(f"Starting scraper for user: {username}")

        # Executa o scraper
        result = await limited_task(execute_scraper, username)
        logger.info(f"Scraper output: {result.stdout}")
        logger.error(f"Scraper errors: {result.stderr}")

        # Localiza o último CSV gerado
        csv_files = sorted(
            [f for f in os.listdir(settings.tweets_folder) if f.endswith('.csv')],
            key=lambda x: os.path.getmtime(os.path.join(settings.tweets_folder, x))
        )
        if not csv_files:
            raise HTTPException(status_code=500, detail="No CSV files found.")

        latest_csv = os.path.join(settings.tweets_folder, csv_files[-1])

        # Valida o CSV e analisa os sentimentos
        df = await limited_task(validate_csv_file, latest_csv)
        results = await limited_task(save_content_and_analyze_sentiment, latest_csv)

        elapsed_time = time.time() - start_time
        logger.info(f"Scraping for user {username} completed in {elapsed_time:.2f} seconds.")

        return {"message": f"Scraping for user {username} completed.", "results": results}
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    is_dev = settings.environment == 'development'
    uvicorn.run("app:app", host=settings.host, port=settings.port, workers=settings.workers, reload=is_dev)