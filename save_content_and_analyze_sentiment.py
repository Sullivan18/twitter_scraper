import sys
import pandas as pd
import re
import json  # Import necessário para salvar o arquivo JSON
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
import warnings
from unidecode import unidecode

# Configuração do logger para salvar em arquivo
logging.basicConfig(
    level=logging.INFO,  # Nível de log (INFO, ERROR, DEBUG etc.)
    filename='analysis.log',  # Nome do arquivo de log
    filemode='w',  # Sobrescreve o arquivo em cada execução
    format='%(asctime)s - %(levellevel)s - %(message)s'  # Formato do log
)

logger = logging.getLogger(__name__)

# Suprimir avisos específicos
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')

# Função para análise de sentimentos e categorias personalizadas
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

# Dicionário expandido de palavras com pesos para categorias de tristeza e felicidade (em português)
custom_dictionary = {
    "tristeza": {
        "triste": -2, "infeliz": -2, "deprimido": -3, "lamentavel": -1, "chateado": -1,
        "desanimado": -2, "abatido": -2, "desolado": -3, "melancolico": -2, "desesperado": -3,
        "derrotado": -2, "angustiado": -2, "sofrimento": -3, "desamparado": -2,
        "lamentavel": -2, "miseravel": -3, "tragedia": -3, "dor": -2, "luto": -3
    },
    "felicidade": {
        "feliz": 2, "alegre": 3, "euforico": 2, "animado": 2, "contente": 1,
        "entusiasmado": 2, "radiante": 3, "satisfeito": 2, "realizado": 2, "extasiado": 3
    }
}

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
        print(results)
