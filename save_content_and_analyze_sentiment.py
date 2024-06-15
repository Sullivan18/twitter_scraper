import sys
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
import warnings

# Configuração do logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suprimir avisos específicos
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')

def preprocess_text(text):
    """Preprocessa o texto para melhorar a análise de sentimento"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()  # Converte para minúsculas
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove menções
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove caracteres especiais
    text = text.strip()  # Remove espaços extras
    return text


def analyze_sentiment(text, tokenizer, model):
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

def save_content_and_analyze_sentiment(input_csv):
    try:
        # Carregar o modelo e o tokenizador
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        logger.info("Model and tokenizer loaded successfully")

        # Ler o CSV de entrada
        df = pd.read_csv(input_csv)
        logger.info(f"Input CSV loaded: {input_csv}")
        
        # Verificar se a coluna 'Content' existe
        if 'Content' not in df.columns:
            logger.error("Input CSV does not contain 'Content' column.")
            return []

        # Adicionar coluna Identifier
        df['Identifier'] = ['tweet{}'.format(i + 1) for i in range(len(df))]

        # Preprocessar o texto e analisar sentimentos
        df['Content'] = df['Content'].apply(lambda x: preprocess_text(x) if pd.notnull(x) else '')
        df[['Sentiment', 'Sentiment_Score']] = df['Content'].apply(
            lambda x: pd.Series(analyze_sentiment(x, tokenizer, model))
        )
        
        # Selecionar apenas as colunas Identifier, Content, Sentiment e Sentiment_Score
        df_content_only = df[['Identifier', 'Content', 'Sentiment', 'Sentiment_Score']]
        
        # Converter para JSON
        results = df_content_only.to_dict(orient='records')
        logger.info("Sentiment analysis completed and converted to JSON")
        return results
    except Exception as e:
        logger.error(f"An error occurred during sentiment analysis: {e}")
        raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python save_content_and_analyze_sentiment.py <input_csv>")
    else:
        input_csv = sys.argv[1]
        results = save_content_and_analyze_sentiment(input_csv)
        print(results)
