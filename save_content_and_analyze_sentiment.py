import sys
import pandas as pd
import re
import json
import logging
import warnings
from unidecode import unidecode
import requests
import os
import json  # Import necessário para salvar o arquivo JSON
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Configuração do logger para salvar em arquivo
logging.basicConfig(
    level=logging.INFO,
    filename='analysis.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Suprimir avisos específicos
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')


def preprocess_text(text):
    """Preprocessa o texto para melhorar a análise de sentimento."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = unidecode(text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

def normalize_word(word):
    """Normaliza uma palavra removendo acentos e convertendo para minúsculas."""
    word = unidecode(word.lower())
    if word.endswith('a') and len(word) > 1:
        word = word[:-1] + 'o'
    return word

def preprocess_csv(df):
    """Preprocessa o DataFrame do CSV para análise."""
    try:
        df = df[['Content', 'Timestamp']].copy()
        df['Content'] = df['Content'].fillna('').str.strip()
        df = df[df['Content'] != '']
        df['Identifier'] = [f'tweet{i+1}' for i in range(len(df))]
        return df
    except KeyError as e:
        logger.error(f"KeyError: {e}")
        raise

def analyze_sentiment(text, tokenizer, model):
    """Realiza a análise de sentimentos"""
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        scores = outputs.logits[0]
        probabilities = torch.nn.functional.softmax(scores, dim=0)
        sentiment = torch.argmax(probabilities).item()
        
        sentiment_labels = ['negative', 'neutral', 'positive']
        sentiment_label = sentiment_labels[sentiment]
        sentiment_score = probabilities[sentiment].item()
        
        return sentiment_label, sentiment_score
    except Exception as e:
        logger.error(f"Error analyzing sentiment for text: {text}. Error: {e}")
        return "error", 0.0
        

custom_dictionary = {
    "tristeza": {
        "triste": -2, "infeliz": -2, "deprimido": -3, "lamentavel": -1, "chateado": -1,
        "desanimado": -2, "abatido": -2, "desolado": -3, "melancolico": -2, "desesperado": -3,
        "derrotado": -2, "angustiado": -2, "sofrimento": -3, "desamparado": -2, "luto": -3,
        "miseravel": -3, "tragedia": -3, "dor": -2, "aflito": -2, "solitario": -2,
        "abandonado": -3, "traição": -3, "pesar": -3, "desesperança": -3, "desgraçado": -3,
        "ansioso": -2, "isolado": -2, "decepcionado": -2, "arrependido": -2, 
        "envergonhado": -2, "amargurado": -2, "injustiçado": -2, "solidao": -3, "perdido": -2,
        "humilhado": -3, "constrangido": -2, "rejeitado": -3, "desprezado": -3, 
        "impotente": -2, "frustrado": -2, "fracassado": -3, "desgostoso": -2, 
        "traiçoeiro": -2, "pesaroso": -2, "ansiedade": -2, "raiva": -2, "tensão": -2, 
        "agonia": -3, "pena": -2, "apatia": -2
    },
    "felicidade": {
        "feliz": 2, "alegre": 3, "euforico": 2, "animado": 2, "contente": 1,
        "entusiasmado": 2, "radiante": 3, "satisfeito": 2, "realizado": 2, "extasiado": 3,
        "grato": 2, "sereno": 2, "aliviado": 1, "apaixonado": 3, "emocionado": 2,
        "otimista": 2, "esperançoso": 2, "abençoado": 3, "encantado": 3, "exultante": 3,
        "afortunado": 2, "felicidade": 3, "paz": 2, "tranquilo": 2, "harmonia": 2,
        "confortável": 2, "próspero": 3, "vitorioso": 3, "brilhante": 2, "empolgado": 2,
        "confiante": 2, "maravilhado": 3, "jubilosamente": 3, "generoso": 2, "divertido": 2,
        "encorajado": 2, "admirado": 2, "conquistador": 3, "positivo": 2, "esperança": 2,
        "fascinado": 2, "deslumbrado": 3, "digno": 2, "satisfeito": 2, "gratidão": 2,
        "orgulhoso": 3, "completo": 2, "inspirado": 3
    }
}


def analyze_custom_category(text, dictionary):
    """Analisa o texto usando um dicionário personalizado."""
    score = 0
    words = text.split()
    for word in words:
        normalized_word = normalize_word(word)
        for category, words_dict in dictionary.items():
            score += words_dict.get(normalized_word, 0)
    
    if score > 0:
        return "felicidade", score
    elif score < 0:
        return "tristeza", score
    else:
        return "neutral", score

def save_content_and_analyze_sentiment(input_csv):
    """Processa o CSV, analisa o sentimento e retorna os resultados"""
    try:
        # Carregamento do modelo e tokenizer apenas uma vez
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        logger.info("Model and tokenizer loaded successfully")
        # Carregar CSV
        df = pd.read_csv(input_csv)
        logger.info(f"Input CSV loaded: {input_csv}")
        
        # Pré-processamento do CSV
        df = preprocess_csv(df)
        logger.info("CSV preprocessed successfully")

        # Análise de sentimentos e categorias personalizadas
        df['Content'] = df['Content'].apply(preprocess_text)
        sentiments = df['Content'].apply(lambda text: analyze_sentiment(text, tokenizer, model))
        custom_categories = df['Content'].apply(lambda text: analyze_custom_category(text, custom_dictionary))
        
        df[['Sentiment', 'Sentiment_Score']] = pd.DataFrame(sentiments.tolist(), index=df.index)
        df[['Custom_Category', 'Custom_Score']] = pd.DataFrame(custom_categories.tolist(), index=df.index)
        
        df_content_only = df[['Identifier', 'Content', 'Timestamp', 'Sentiment', 'Sentiment_Score', 'Custom_Category', 'Custom_Score']]
        
        results = df_content_only.to_dict(orient='records')
        logger.info("Sentiment analysis completed and converted to JSON")

        # **Salvando o JSON em arquivo**
        json_filename = "./sentiment_json/sentiment_analysis_results.json"  # Nome do arquivo JSON
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=4)
        logger.info(f"Results saved to {json_filename}")

        return results
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"EmptyDataError: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during sentiment analysis: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python save_content_and_analyze_sentiment.py <input_csv>")
    else:
        input_csv = sys.argv[1]
        results = save_content_and_analyze_sentiment(input_csv)