import sys
import pandas as pd
import re
import json
import logging
import warnings
from unidecode import unidecode
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime


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
    """Preprocessa o texto para melhorar a análise de sentimento"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = unidecode(text)  # Remove acentos
    text = re.sub(r'[^\w\s]', '', text)  # Remove pontuação
    text = text.strip()
    return text


def normalize_word(word):
    """Normaliza a palavra removendo acentos, convertendo para minúsculas e lematizando"""
    word = unidecode(word.lower())
    if word.endswith('a') and len(word) > 1:
        word = word[:-1] + 'o'  # Lematização simples
    return word


def preprocess_csv(df):
    """Preprocessa o DataFrame do CSV para análise de sentimento"""
    try:
        df = df[['Content', 'Timestamp']].copy()
        df['Content'].fillna('', inplace=True)
        df['Identifier'] = [f'tweet{i+1}' for i in range(len(df))]
        return df
    except KeyError as e:
        logger.error(f"KeyError in preprocessing CSV: {e}")
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


def load_custom_dictionary_from_excel(file_path):
    """Carrega um dicionário customizado a partir de um arquivo Excel"""
    try:
        xls = pd.ExcelFile(file_path)
        custom_dict = {}

        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            if sheet.upper() in df.columns and 'VALOR' in df.columns:
                category_dict = pd.Series(df['VALOR'].values, index=df[sheet.upper()]).to_dict()
                custom_dict[sheet.lower()] = category_dict
            else:
                logger.warning(f"Aba {sheet} não contém as colunas esperadas ('{sheet.upper()}', 'VALOR'). Ignorada.")

        logger.info(f"Dicionário customizado carregado com sucesso do arquivo: {file_path}")
        return custom_dict
    except FileNotFoundError:
        logger.error(f"Arquivo Excel não encontrado: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Erro ao carregar o dicionário customizado do Excel: {e}")
        raise


def analyze_custom_category(text, dictionary):
    """Analisa o texto usando um dicionário personalizado e retorna a categoria"""
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


def save_content_and_analyze_sentiment(input_csv, dictionary_excel_path):
    """Processa o CSV e analisa o sentimento usando o dicionário customizado"""
    try:
        # Carregamento do modelo e tokenizer
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        logger.info("Model and tokenizer loaded successfully")

        # Capturar o timestamp da análise
        analysis_timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.000Z')

        # Carregar o dicionário customizado do Excel
        custom_dictionary = load_custom_dictionary_from_excel(dictionary_excel_path)

        # Carregar CSV
        df_csv = pd.read_csv(input_csv)
        logger.info(f"Input CSV loaded: {input_csv}")

        # Pré-processar o CSV
        df_csv = preprocess_csv(df_csv)
        logger.info("CSV preprocessed successfully")

        # Pré-processar o conteúdo do CSV
        df_csv['Content'] = df_csv['Content'].apply(preprocess_text)

        # Analisar sentimentos e categorias personalizadas
        sentiments_csv = df_csv['Content'].apply(lambda text: analyze_sentiment(text, tokenizer, model))
        custom_categories_csv = df_csv['Content'].apply(lambda text: analyze_custom_category(text, custom_dictionary))

        df_csv[['Sentiment', 'Sentiment_Score']] = pd.DataFrame(sentiments_csv.tolist(), index=df_csv.index)
        df_csv[['Custom_Category', 'Custom_Score']] = pd.DataFrame(custom_categories_csv.tolist(), index=df_csv.index)

        df_csv_content_only = df_csv[['Identifier', 'Content', 'Timestamp', 'Sentiment', 'Sentiment_Score', 'Custom_Category', 'Custom_Score']]

        # Adicionar a data da análise a cada registro
        df_csv_content_only['Analysis_Timestamp'] = analysis_timestamp

        # Combinar os resultados
        results = {
            'analysis_timestamp': analysis_timestamp,  # Timestamp da análise geral
            'tweets_analysis': df_csv_content_only.to_dict(orient='records')
        }

        # Salvar resultados em JSON
        json_filename = "./sentiment_json/sentiment_analysis_results.json"
        os.makedirs(os.path.dirname(json_filename), exist_ok=True)
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=4)
        logger.info(f"Results saved to {json_filename}")

        return results
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise



if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.error("Usage: python save_content_and_analyze_sentiment.py <input_csv> <dictionary_excel_path>")
    else:
        input_csv = sys.argv[1]
        dictionary_excel_path = sys.argv[2]
        save_content_and_analyze_sentiment(input_csv, dictionary_excel_path)
