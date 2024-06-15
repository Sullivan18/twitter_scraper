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

def preprocess_text(text):
    """Preprocessa o texto para melhorar a análise de sentimento"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()  # Converte para minúsculas
    text = unidecode(text)  # Remove acentuação
    text = re.sub(r'[^\w\s]', '', text)  # Remove pontuações
    text = text.strip()  # Remove espaços extras
    return text

def normalize_word(word):
    """Normaliza a palavra removendo acentos, convertendo para minúsculas e lematizando"""
    word = unidecode(word.lower())
    if word.endswith('a') and len(word) > 1:
        word = word[:-1] + 'o'  # Transforma palavras femininas para masculinas
    return word

def preprocess_csv(df):
    """Preprocessa o DataFrame do CSV para análise de sentimento"""
    # Selecionar apenas a coluna 'Content'
    df = df[['Content']].copy()
    
    # Preencher valores nulos
    df.loc[:, 'Content'] = df['Content'].fillna('')
    
    # Adicionar coluna Identifier
    df.loc[:, 'Identifier'] = ['tweet{}'.format(i + 1) for i in range(len(df))]
    
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
    logger.info(f"Analyzing custom category for text: {text}")
    score = 0
    words = text.split()
    for word in words:
        clean_word = re.sub(r'[^\w\s]', '', word)
        normalized_word = normalize_word(clean_word)
        logger.info(f"Checking word: {clean_word} (normalized: {normalized_word})")
        for category, words_dict in dictionary.items():
            if normalized_word in words_dict:
                logger.info(f"Found word '{normalized_word}' in category '{category}' with score {words_dict[normalized_word]}")
                score += words_dict[normalized_word]
            else:
                logger.info(f"Word '{normalized_word}' not found in category '{category}'")
    
    logger.info(f"Total custom score for text: {score}")
    
    if score > 0:
        return "felicidade", score
    elif score < 0:
        return "tristeza", score
    else:
        return "neutral", score

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
        
        # Pré-processar o DataFrame
        df = preprocess_csv(df)
        logger.info("CSV preprocessed successfully")

        # Preprocessar o texto e analisar sentimentos
        df['Content'] = df['Content'].apply(lambda x: preprocess_text(x) if pd.notnull(x) else '')
        df[['Sentiment', 'Sentiment_Score']] = df['Content'].apply(
            lambda x: pd.Series(analyze_sentiment(x, tokenizer, model))
        )
        df[['Custom_Category', 'Custom_Score']] = df['Content'].apply(
            lambda x: pd.Series(analyze_custom_category(x, custom_dictionary))
        )
        
        # Selecionar apenas as colunas Identifier, Content, Sentiment, Sentiment_Score, Custom_Category e Custom_Score
        df_content_only = df[['Identifier', 'Content', 'Sentiment', 'Sentiment_Score', 'Custom_Category', 'Custom_Score']]
        
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
