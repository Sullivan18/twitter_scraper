import sys
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
import warnings
from unidecode import unidecode


# Configuração do logger para salvar em arquivo
logging.basicConfig(level=logging.INFO, filename='analysis.log', filemode='w', format='%(asctime)s - %(levellevel)s - %(message)s')
logger = logging.getLogger(__name__)

# Suprimir avisos específicos
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')

# Função para análise de sentimentos e categorias personalizadas
def preprocess_text(text):
    """Preprocessa o texto para melhorar a análise de sentimento"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = unidecode(text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

def normalize_word(word):
    """Normaliza a palavra removendo acentos, convertendo para minúsculas e lematizando"""
    word = unidecode(word.lower())
    if word.endswith('a') and len(word) > 1:
        word = word[:-1] + 'o'
    return word

def preprocess_csv(df):
    """Preprocessa o DataFrame do CSV para análise de sentimento"""
    df = df[['Content', 'Timestamp']].copy()
    df['Content'] = df['Content'].fillna('')
    df['Identifier'] = ['tweet{}'.format(i + 1) for i in range(len(df))]
    return df

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

# Dicionário expandido de palavras com pesos para categorias de tristeza e felicidade (em português)
custom_dictionary = {
    "tristeza": {
        "triste": -2, "infeliz": -2, "deprimido": -3, "lamentavel": -1, "chateado": -1,
        "desanimado": -2, "abatido": -2, "desolado": -3, "melancolico": -2, "desesperado": -3,
        "desesperanca": -3, "derrotado": -2, "angustiado": -2, "sofrimento": -3, "desamparado": -2,
        "lamentavel": -2, "miseravel": -3, "tragedia": -3, "dor": -2, "luto": -3,
        "sofrido": -3, "solitario": -2, "desprezado": -2, "desgostoso": -2, "aflito": -2,
        "amedrontado": -2, "ansioso": -2, "arrependido": -2, "confuso": -1, "culpado": -2,
        "desesperanca": -3, "desprezivel": -2, "duvida": -1, "inseguro": -2, "insignificante": -2,
        "perdido": -2, "remorso": -2, "saudade": -2, "vulneravel": -2, "humilhado": -3,
        "derrotado": -2, "abatido": -2, "inutil": -2, "pessimo": -2, "arrasado": -3,
        "desencorajado": -2, "inconsolavel": -3, "oprimido": -3, "excluido": -2, "depressivo": -3
    },
    "felicidade": {
        "feliz": 2, "alegre": 3, "euforico": 2, "animado": 2, "contente": 1,
        "entusiasmado": 2, "radiante": 3, "satisfeito": 2, "realizado": 2, "extasiado": 3,
        "empolgado": 2, "triumfante": 3, "grato": 2, "positivo": 2, "exultante": 3,
        "alegria": 3, "sorridente": 2, "felicidade": 3, "bem-estar": 2, "abençoado": 2,
        "afortunado": 2, "divertido": 2, "divertido": 2, "animador": 2, "euforia": 3,
        "satisfeito": 2, "prazer": 2, "contentamento": 2, "encantado": 3, "enlevado": 3,
        "extase": 3, "exuberante": 2, "alegremente": 3, "sorriso": 2, "bemaventurado": 2,
        "agradavel": 2, "fantastico": 2, "otimista": 2, "maravilhoso": 2, "inspirador": 2,
        "emocionado": 2, "sereno": 2, "paz": 2, "sucesso": 3, "vitoria": 3,
        "afabilidade": 2, "bem-humorado": 2, "extremamente feliz": 3, "gozoso": 2, "triumfo": 3,
        "vivaz": 2, "admiravel": 2, "incrivel": 2, "formidavel": 2, "sensacional": 3
    }
}


def analyze_custom_category(text, dictionary):
    """Analisa o texto usando um dicionário personalizado e retorna a categoria"""
    score = 0
    words = text.split()
    for word in words:
        normalized_word = normalize_word(word)
        for category, words_dict in dictionary.items():
            if normalized_word in words_dict:
                score += words_dict[normalized_word]
    
    if score > 0:
        return "felicidade", score
    elif score < 0:
        return "tristeza", score
    else:
        return "neutral", score

def save_content_and_analyze_sentiment(input_csv):
    try:
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        logger.info("Model and tokenizer loaded successfully")

        df = pd.read_csv(input_csv)
        logger.info(f"Input CSV loaded: {input_csv}")
        
        df = preprocess_csv(df)
        logger.info("CSV preprocessed successfully")

        df['Content'] = df['Content'].apply(lambda x: preprocess_text(x) if pd.notnull(x) else '')
        sentiments = [analyze_sentiment(text, tokenizer, model) for text in df['Content']]
        custom_categories = [analyze_custom_category(text, custom_dictionary) for text in df['Content']]
        
        df[['Sentiment', 'Sentiment_Score']] = pd.DataFrame(sentiments, index=df.index)
        df[['Custom_Category', 'Custom_Score']] = pd.DataFrame(custom_categories, index=df.index)
        
        df_content_only = df[['Identifier', 'Content', 'Timestamp', 'Sentiment', 'Sentiment_Score', 'Custom_Category', 'Custom_Score']]
        
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