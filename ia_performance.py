# Importar as bibliotecas necessárias
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Carregar o dataset
file_path = './dataset/Test3classes.csv'  # Atualize o caminho conforme necessário
dataset = pd.read_csv(file_path, sep=';')

# Função para carregar o modelo e o tokenizer
def load_model_and_tokenizer():
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Função para pré-processar os tweets
def preprocess_tweet(tweet, tokenizer, max_length=128):
    return tokenizer(tweet, truncation=True, max_length=max_length, padding='max_length', return_tensors='pt')

# Função para realizar a predição de um tweet
def predict_sentiment(tweet, tokenizer, model, labels=['Negative', 'Neutral', 'Positive']):
    inputs = preprocess_tweet(tweet, tokenizer)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        return labels[predicted_class]

# Função para calcular as métricas de performance do modelo
def calculate_performance_metrics(dataset_tweets, true_labels, tokenizer, model):
    predicted_labels = []
    
    for tweet in dataset_tweets:
        predicted_sentiment = predict_sentiment(tweet, tokenizer, model)
        predicted_labels.append(predicted_sentiment)

    # Calcular as métricas
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    
    # Gerar o relatório completo de métricas
    report = classification_report(true_labels, predicted_labels, target_names=['Negative', 'Neutral', 'Positive'])

    return accuracy, precision, recall, f1, report

# Converter os rótulos do dataset para as classes de sentimento correspondentes
label_map = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
dataset['sentiment_label'] = dataset['sentiment'].map(label_map)

# Carregar o modelo e tokenizer
tokenizer, model = load_model_and_tokenizer()

# Calcular as métricas de performance com base no dataset
tweets = dataset['tweet_text'].tolist()
true_labels = dataset['sentiment_label'].tolist()
accuracy, precision, recall, f1, report = calculate_performance_metrics(tweets, true_labels, tokenizer, model)

# Exibir os resultados
print(f"Acurácia: {accuracy:.2f}")
print(f"Precisão: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print("\nRelatório de Classificação:\n", report)
