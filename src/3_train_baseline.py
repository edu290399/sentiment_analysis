"""
Treinamento de modelo baseline para análise de sentimentos
Pipeline: TF-IDF + Logistic Regression
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from tqdm import tqdm

# Importar função de preprocessamento do módulo local
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar do arquivo 2_preprocessing.py
import importlib.util
spec = importlib.util.spec_from_file_location("preprocessing", 
                                               os.path.join(os.path.dirname(__file__), "2_preprocessing.py"))
preprocessing_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocessing_module)
clean_text = preprocessing_module.clean_text


def load_data(file_path='data/sentiment140_processed.csv'):
    """
    Carrega os dados processados.
    
    Args:
        file_path (str): Caminho do arquivo CSV
    
    Returns:
        pd.DataFrame: DataFrame com os dados
    """
    print(f"Carregando dados de: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"✓ Dados carregados! Shape: {df.shape}")
    
    return df


def preprocess_texts(df, text_column='text'):
    """
    Aplica a função clean_text em todos os textos.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
        text_column (str): Nome da coluna de texto
    
    Returns:
        pd.DataFrame: DataFrame com coluna de texto limpo
    """
    print(f"\nAplicando preprocessamento em {len(df)} textos...")
    
    # Aplicar clean_text com barra de progresso
    tqdm.pandas(desc="Limpando textos")
    df['text_clean'] = df[text_column].progress_apply(clean_text)
    
    # Remover textos vazios após limpeza
    original_size = len(df)
    df = df[df['text_clean'].str.strip() != '']
    removed = original_size - len(df)
    
    print(f"✓ Preprocessamento concluído!")
    print(f"  - Textos removidos (vazios após limpeza): {removed}")
    print(f"  - Shape final: {df.shape}")
    
    return df


def create_pipeline():
    """
    Cria um pipeline de ML com TF-IDF e Logistic Regression.
    
    Returns:
        Pipeline: Pipeline do scikit-learn
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,      # Número máximo de features
            ngram_range=(1, 2),      # Unigramas e bigramas
            min_df=2,                # Ignorar termos que aparecem em menos de 2 documentos
            max_df=0.95,             # Ignorar termos que aparecem em mais de 95% dos documentos
            sublinear_tf=True        # Aplicar escala logarítmica à frequência dos termos
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0,                   # Regularização
            solver='lbfgs',
            class_weight='balanced'  # Balancear classes automaticamente
        ))
    ])
    
    return pipeline


def train_baseline_model(X_train, y_train, X_test, y_test):
    """
    Treina o modelo baseline e avalia sua performance.
    
    Args:
        X_train: Textos de treino
        y_train: Labels de treino
        X_test: Textos de teste
        y_test: Labels de teste
    
    Returns:
        Pipeline: Pipeline treinado
    """
    print("\n" + "="*70)
    print("TREINAMENTO DO MODELO BASELINE")
    print("="*70)
    
    # Criar pipeline
    print("\nCriando pipeline: TfidfVectorizer + LogisticRegression")
    pipeline = create_pipeline()
    
    # Treinar modelo
    print(f"\nTreinando modelo com {len(X_train)} exemplos...")
    pipeline.fit(X_train, y_train)
    print("✓ Treinamento concluído!")
    
    # Fazer predições
    print("\nFazendo predições no conjunto de teste...")
    y_pred = pipeline.predict(X_test)
    
    # Avaliar modelo
    print("\n" + "="*70)
    print("AVALIAÇÃO DO MODELO")
    print("="*70)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification Report
    print("\nClassification Report:")
    print("-" * 70)
    target_names = ['Negativo (0)', 'Positivo (1)']
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))
    
    # Confusion Matrix
    print("Confusion Matrix:")
    print("-" * 70)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nInterpretação da Matriz de Confusão:")
    print(f"  True Negatives:  {cm[0][0]:,} | False Positives: {cm[0][1]:,}")
    print(f"  False Negatives: {cm[1][0]:,} | True Positives:  {cm[1][1]:,}")
    
    # Informações adicionais sobre o modelo
    print("\n" + "="*70)
    print("INFORMAÇÕES DO MODELO")
    print("="*70)
    
    tfidf = pipeline.named_steps['tfidf']
    print(f"\nTF-IDF Vectorizer:")
    print(f"  - Vocabulário: {len(tfidf.vocabulary_):,} termos únicos")
    print(f"  - N-grams: {tfidf.ngram_range}")
    print(f"  - Max features: {tfidf.max_features:,}")
    
    clf = pipeline.named_steps['clf']
    print(f"\nLogistic Regression:")
    print(f"  - Coeficientes: {clf.coef_.shape}")
    print(f"  - Iterações: {clf.n_iter_}")
    
    return pipeline


def save_model(pipeline, model_path='models/baseline_model.pkl'):
    """
    Salva o pipeline treinado.
    
    Args:
        pipeline: Pipeline treinado
        model_path (str): Caminho para salvar o modelo
    """
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Salvar modelo
    joblib.dump(pipeline, model_path)
    print(f"\n✓ Modelo salvo em: {model_path}")
    
    # Informar tamanho do arquivo
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  Tamanho do arquivo: {size_mb:.2f} MB")


def load_model(model_path='models/baseline_model.pkl'):
    """
    Carrega um pipeline treinado.
    
    Args:
        model_path (str): Caminho do modelo salvo
    
    Returns:
        Pipeline: Pipeline carregado
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
    pipeline = joblib.load(model_path)
    print(f"✓ Modelo carregado de: {model_path}")
    
    return pipeline


def predict_sentiment(pipeline, texts):
    """
    Faz predições em novos textos.
    
    Args:
        pipeline: Pipeline treinado
        texts (list): Lista de textos
    
    Returns:
        np.array: Predições
    """
    # Limpar textos
    cleaned_texts = [clean_text(text) for text in texts]
    
    # Fazer predições
    predictions = pipeline.predict(cleaned_texts)
    probabilities = pipeline.predict_proba(cleaned_texts)
    
    return predictions, probabilities


def main():
    """
    Função principal para executar o treinamento do modelo baseline.
    """
    print("="*70)
    print("ANÁLISE DE SENTIMENTOS - MODELO BASELINE")
    print("Pipeline: TF-IDF + Logistic Regression")
    print("="*70)
    
    # 1. Carregar dados
    df = load_data('data/sentiment140_processed.csv')
    
    # Mostrar distribuição de classes
    print("\nDistribuição de classes:")
    print(df['label'].value_counts().sort_index())
    
    # 2. Aplicar preprocessamento
    df = preprocess_texts(df, text_column='text')
    
    # 3. Separar features e target
    X = df['text_clean'].values
    y = df['label'].values
    
    print(f"\nDataset preparado:")
    print(f"  - Total de exemplos: {len(X):,}")
    print(f"  - Classes: {np.unique(y)}")
    
    # 4. Dividir em treino e teste (80/20)
    print("\nDividindo dados em treino (80%) e teste (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y  # Manter proporção de classes
    )
    
    print(f"✓ Split realizado:")
    print(f"  - Treino: {len(X_train):,} exemplos")
    print(f"  - Teste:  {len(X_test):,} exemplos")
    
    # 5. Treinar modelo
    pipeline = train_baseline_model(X_train, y_train, X_test, y_test)
    
    # 6. Salvar modelo
    save_model(pipeline)
    
    # 7. Testar predições em exemplos
    print("\n" + "="*70)
    print("TESTE DE PREDIÇÕES")
    print("="*70)
    
    test_examples = [
        "I love this product! It's amazing and works perfectly!",
        "This is the worst experience ever. Terrible service.",
        "Not bad, but could be better. Average quality.",
        "Absolutely fantastic! Best purchase I've ever made!",
        "Disappointed with the quality. Would not recommend."
    ]
    
    predictions, probabilities = predict_sentiment(pipeline, test_examples)
    
    print("\nExemplos de predições:")
    for i, (text, pred, prob) in enumerate(zip(test_examples, predictions, probabilities), 1):
        sentiment = "POSITIVO" if pred == 1 else "NEGATIVO"
        confidence = prob[pred] * 100
        print(f"\n{i}. Texto: {text[:60]}...")
        print(f"   Predição: {sentiment} (confiança: {confidence:.2f}%)")
    
    print("\n" + "="*70)
    print("✓ TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("="*70)


if __name__ == "__main__":
    main()
