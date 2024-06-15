import sys
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def analyze_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    scores = outputs.logits[0]
    probabilities = torch.nn.functional.softmax(scores, dim=0)
    sentiment = torch.argmax(probabilities).item()
    
    sentiment_labels = ['negative', 'neutral', 'positive']
    sentiment_label = sentiment_labels[sentiment]
    sentiment_score = probabilities[sentiment].item()
    
    return sentiment_label, sentiment_score

def save_content_and_analyze_sentiment(input_csv, output_csv):
    # Carregar o modelo e o tokenizador
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Ler o CSV de entrada
    df = pd.read_csv(input_csv)
    
    # Adicionar coluna Identifier
    df['Identifier'] = ['tweet{}'.format(i + 1) for i in range(len(df))]
    
    # Analisar sentimentos
    df[['Sentiment', 'Sentiment_Score']] = df['Content'].apply(
        lambda x: pd.Series(analyze_sentiment(x, tokenizer, model))
    )
    
    # Selecionar apenas as colunas Identifier, Content, Sentiment e Sentiment_Score
    df_content_only = df[['Identifier', 'Content', 'Sentiment', 'Sentiment_Score']]
    
    # Salvar o CSV de saída
    df_content_only.to_csv(output_csv, index=False, encoding='utf-8')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python save_content_and_analyze_sentiment.py <input_csv> <output_csv>")
    else:
        input_csv = sys.argv[1]
        output_csv = sys.argv[2]
        save_content_and_analyze_sentiment(input_csv, output_csv)
