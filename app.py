from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os
from datetime import datetime

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
        # Execute o comando do scraper
        command = f"python scraper -t 10 -u {username}"
        subprocess.run(command, shell=True)
        
        # Localizar o último CSV gerado na pasta de tweets
        tweets_folder = './tweets/'
        csv_files = sorted([f for f in os.listdir(tweets_folder) if f.endswith('.csv')], key=lambda x: os.path.getmtime(os.path.join(tweets_folder, x)))
        if not csv_files:
            raise HTTPException(status_code=500, detail="No CSV files found.")
        
        latest_csv = os.path.join(tweets_folder, csv_files[-1])
        output_csv = os.path.join(tweets_folder, f'content_only_tweets_{username}.csv')
        
        # Chama o script save_content_only.py
        extract_command = f"python save_content_only.py {latest_csv} {output_csv}"
        subprocess.run(extract_command, shell=True)
        
        # Remove o CSV original
        os.remove(latest_csv)
        
        return {"message": f"Scraping for user {username} started. Content saved to {output_csv}"}
    raise HTTPException(status_code=400, detail="Username is required.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
