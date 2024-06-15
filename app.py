from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os
import logging

# Importar a função save_content_and_analyze_sentiment
from save_content_and_analyze_sentiment import save_content_and_analyze_sentiment

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class UserRequest(BaseModel):
    username: str

@app.get("/")
def read_root():
    return {"message": "API is working. Use the /scrape endpoint to start scraping."}

@app.post("/scrape")
def scrape(user_request: UserRequest):
    username = user_request.username
    if username:
        try:
            logger.info(f"Starting scraper for user: {username}")
            # Execute o comando do scraper (simulando a execução de um script)
            command = f"python scraper -t 100 -u {username}"
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            logger.info(f"Scraper output: {result.stdout}")
            logger.error(f"Scraper errors: {result.stderr}")

            # Localizar o último CSV gerado na pasta de tweets
            tweets_folder = './tweets/'
            csv_files = sorted([f for f in os.listdir(tweets_folder) if f.endswith('.csv')], key=lambda x: os.path.getmtime(os.path.join(tweets_folder, x)))
            if not csv_files:
                raise HTTPException(status_code=500, detail="No CSV files found.")
            
            latest_csv = os.path.join(tweets_folder, csv_files[-1])
            
            # Chama a função save_content_and_analyze_sentiment
            results = save_content_and_analyze_sentiment(latest_csv)
            
            return {"message": f"Scraping for user {username} completed.", "results": results}
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(status_code=400, detail="Username is required.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
