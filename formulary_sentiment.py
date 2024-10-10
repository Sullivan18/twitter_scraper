import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
import time

# Função auxiliar para registrar o tempo decorrido
def record_time(start_time):
    return time.time() - start_time

# Dicionário para armazenar os tempos de execução
execution_times = {}

# Medir tempo total de execução
total_start_time = time.time()

# 1. Carregar o arquivo Excel
start_time = time.time()
df = pd.read_excel('COMO EU ME VEJO.xlsx')
execution_times["Carregar arquivo Excel"] = record_time(start_time)

# 2. Carregar o modelo e tokenizer do Hugging Face
start_time = time.time()
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
execution_times["Carregar modelo e tokenizer"] = record_time(start_time)

# 3. Configurar o pipeline de análise de sentimento
start_time = time.time()
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
execution_times["Configurar pipeline de análise de sentimento"] = record_time(start_time)

# 4. Aplicar a análise de sentimento nas respostas dos alunos
start_time = time.time()
responses = df["Resposta Aluno"].astype(str).tolist()
sentiment_results = sentiment_analyzer(responses)
execution_times["Aplicar análise de sentimento nas respostas"] = record_time(start_time)

# 5. Armazenar os resultados da análise de sentimento
start_time = time.time()
df["Sentiment"] = [result['label'] for result in sentiment_results]
execution_times["Armazenar resultados da análise de sentimento"] = record_time(start_time)

# 6. INSIGHT 1: Distribuição de Sentimentos por Questão
start_time = time.time()
sentiment_by_question = df.groupby("Numero da Questão")["Sentiment"].value_counts().unstack(fill_value=0)
execution_times["Calcular distribuição de sentimentos por questão"] = record_time(start_time)

# 7. INSIGHT 2: Sentimentos por Semestre e Turno
start_time = time.time()
sentiment_by_semester = df.groupby(["Semestre", "Turno"])["Sentiment"].value_counts().unstack(fill_value=0)
execution_times["Calcular sentimentos por semestre e turno"] = record_time(start_time)

# 8. INSIGHT 5: Tendência Temporal de Sentimentos
start_time = time.time()
df['Data Realização'] = pd.to_datetime(df['Data Realização'], errors='coerce')
df['Month'] = df['Data Realização'].dt.to_period('M')
sentiment_by_time = df.groupby("Month")["Sentiment"].value_counts().unstack(fill_value=0)
execution_times["Calcular tendência temporal de sentimentos"] = record_time(start_time)

# Exibir resultados de análise
print("Distribuição de Sentimentos por Questão:")
print(sentiment_by_question)
print("\nDistribuição de Sentimentos por Semestre e Turno:")
print(sentiment_by_semester)
print("\nTendência Temporal de Sentimentos:")
print(sentiment_by_time)

# Exibir os tempos de execução separados (removendo os tempos zerados)
print("\nTempos de execução de cada tarefa (em segundos):")
for task, duration in execution_times.items():
    if duration > 0:
        print(f"{task}: {duration:.5f} segundos")

# Tempo total de execução
total_time = time.time() - total_start_time
print(f"\nTempo total de execução: {total_time:.5f} segundos")
