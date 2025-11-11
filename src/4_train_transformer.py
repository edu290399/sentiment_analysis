"""
Fine-tuning de modelo Transformer para Análise de Sentimentos
Modelo: DistilBERT-base-uncased
Dataset: Sentiment140
"""

import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score


# Configurações
MODEL_NAME = "distilbert-base-uncased"  # Modelo base eficiente
NUM_LABELS = 2  # Classificação binária: 0 (Negativo), 1 (Positivo)
MAX_LENGTH = 128  # Comprimento máximo dos tweets
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01


def load_data(file_path='data/sentiment140_processed.csv', sample_size=None):
    """
    Carrega os dados processados do Sentiment140.
    
    Args:
        file_path (str): Caminho do arquivo CSV
        sample_size (int): Número de exemplos a carregar (None = todos)
    
    Returns:
        pd.DataFrame: DataFrame com os dados
    """
    print("="*70)
    print("CARREGANDO DADOS")
    print("="*70)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    print(f"\nCarregando dados de: {file_path}")
    df = pd.read_csv(file_path)
    
    if sample_size and sample_size < len(df):
        print(f"Amostrando {sample_size:,} exemplos do total de {len(df):,}")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    print(f"✓ Dados carregados! Shape: {df.shape}")
    print(f"\nDistribuição de classes:")
    print(df['label'].value_counts().sort_index())
    
    return df


def prepare_datasets(df, test_size=0.2):
    """
    Divide os dados em treino e validação e converte para Dataset do Hugging Face.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
        test_size (float): Proporção do conjunto de validação
    
    Returns:
        tuple: (train_dataset, eval_dataset)
    """
    print("\n" + "="*70)
    print("PREPARANDO DATASETS")
    print("="*70)
    
    # Dividir em treino e validação
    print(f"\nDividindo dados: {int((1-test_size)*100)}% treino / {int(test_size*100)}% validação")
    
    train_df, eval_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df['label']  # Manter proporção de classes
    )
    
    print(f"✓ Treino: {len(train_df):,} exemplos")
    print(f"✓ Validação: {len(eval_df):,} exemplos")
    
    # Converter para Dataset do Hugging Face
    print("\nConvertendo para formato Dataset...")
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    eval_dataset = Dataset.from_pandas(eval_df[['text', 'label']])
    
    print("✓ Conversão concluída!")
    
    return train_dataset, eval_dataset


def tokenize_function(examples, tokenizer):
    """
    Tokeniza os textos usando o tokenizer do modelo.
    
    Args:
        examples: Batch de exemplos do dataset
        tokenizer: Tokenizer do Hugging Face
    
    Returns:
        dict: Textos tokenizados
    """
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH
    )


def tokenize_datasets(train_dataset, eval_dataset, tokenizer):
    """
    Aplica tokenização aos datasets de treino e validação.
    
    Args:
        train_dataset: Dataset de treino
        eval_dataset: Dataset de validação
        tokenizer: Tokenizer do Hugging Face
    
    Returns:
        tuple: (train_dataset_tokenized, eval_dataset_tokenized)
    """
    print("\n" + "="*70)
    print("TOKENIZAÇÃO DOS DADOS")
    print("="*70)
    
    print("\nTokenizando conjunto de treino...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        desc="Tokenizando treino"
    )
    
    print("Tokenizando conjunto de validação...")
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        desc="Tokenizando validação"
    )
    
    print("✓ Tokenização concluída!")
    
    return train_dataset, eval_dataset


def compute_metrics(eval_pred):
    """
    Calcula métricas de avaliação para o Trainer.
    
    Args:
        eval_pred: Tupla (predictions, labels)
    
    Returns:
        dict: Dicionário com métricas
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calcular métricas
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def create_trainer(model, tokenizer, train_dataset, eval_dataset):
    """
    Cria e configura o Trainer do Hugging Face.
    
    Args:
        model: Modelo para treinamento
        tokenizer: Tokenizer
        train_dataset: Dataset de treino tokenizado
        eval_dataset: Dataset de validação tokenizado
    
    Returns:
        Trainer: Trainer configurado
    """
    print("\n" + "="*70)
    print("CONFIGURANDO TREINAMENTO")
    print("="*70)
    
    # Definir argumentos de treinamento
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        save_total_limit=2,  # Manter apenas os 2 melhores checkpoints
        fp16=torch.cuda.is_available(),  # Mixed precision se GPU disponível
        report_to='none'  # Desabilitar integrações (wandb, tensorboard, etc)
    )
    
    print("\nConfiguração do Treinamento:")
    print(f"  - Modelo: {MODEL_NAME}")
    print(f"  - Épocas: {NUM_EPOCHS}")
    print(f"  - Batch size (treino): {TRAIN_BATCH_SIZE}")
    print(f"  - Batch size (validação): {EVAL_BATCH_SIZE}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Warmup steps: {WARMUP_STEPS}")
    print(f"  - Weight decay: {WEIGHT_DECAY}")
    print(f"  - Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    print(f"  - Mixed Precision (FP16): {torch.cuda.is_available()}")
    
    # Data collator para padding dinâmico
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Criar Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    print("\n✓ Trainer configurado!")
    
    return trainer


def train_model(trainer):
    """
    Executa o treinamento do modelo.
    
    Args:
        trainer: Trainer configurado
    
    Returns:
        dict: Métricas de treinamento
    """
    print("\n" + "="*70)
    print("INICIANDO TREINAMENTO")
    print("="*70)
    print("\nIsso pode levar vários minutos dependendo do hardware...")
    print("Progresso será mostrado abaixo:\n")
    
    # Treinar modelo
    train_result = trainer.train()
    
    print("\n✓ Treinamento concluído!")
    
    return train_result


def evaluate_model(trainer):
    """
    Avalia o modelo no conjunto de validação.
    
    Args:
        trainer: Trainer com modelo treinado
    
    Returns:
        dict: Métricas de avaliação
    """
    print("\n" + "="*70)
    print("AVALIAÇÃO FINAL")
    print("="*70)
    
    print("\nAvaliando modelo no conjunto de validação...")
    eval_results = trainer.evaluate()
    
    print("\n✓ Avaliação concluída!")
    print("\nResultados Finais:")
    print("-" * 70)
    for metric, value in eval_results.items():
        if metric.startswith('eval_'):
            metric_name = metric.replace('eval_', '').upper()
            if 'loss' in metric:
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value:.4f} ({value*100:.2f}%)")
    
    return eval_results


def save_model(trainer, tokenizer, save_dir="./models/sentiment_transformer"):
    """
    Salva o modelo treinado e o tokenizer.
    
    Args:
        trainer: Trainer com modelo treinado
        tokenizer: Tokenizer
        save_dir (str): Diretório para salvar
    """
    print("\n" + "="*70)
    print("SALVANDO MODELO")
    print("="*70)
    
    # Criar diretório se não existir
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nSalvando modelo e tokenizer em: {save_dir}")
    
    # Salvar modelo e tokenizer
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    print("✓ Modelo salvo com sucesso!")
    
    # Informar tamanho
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(save_dir)
        for filename in filenames
    )
    print(f"  Tamanho total: {total_size / (1024**2):.2f} MB")


def test_predictions(model, tokenizer, device):
    """
    Testa o modelo com exemplos de predição.
    
    Args:
        model: Modelo treinado
        tokenizer: Tokenizer
        device: Device (cuda/cpu)
    """
    print("\n" + "="*70)
    print("TESTE DE PREDIÇÕES")
    print("="*70)
    
    # Exemplos de teste
    test_texts = [
        "I absolutely love this product! It's amazing and works perfectly!",
        "This is the worst experience I've ever had. Terrible service.",
        "Great quality and fast delivery. Highly recommend!",
        "Disappointed with the purchase. Not worth the money.",
        "It's okay, nothing special but does the job.",
        "Best decision ever! So happy with this purchase!",
        "Waste of time and money. Very disappointed.",
        "Pretty good overall. Met my expectations."
    ]
    
    model.eval()
    
    print("\nExemplos de predições:")
    print("-" * 70)
    
    for i, text in enumerate(test_texts, 1):
        # Tokenizar
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=MAX_LENGTH,
            padding='max_length'
        ).to(device)
        
        # Predição
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][prediction].item()
        
        sentiment = "POSITIVO ✅" if prediction == 1 else "NEGATIVO ❌"
        
        print(f"\n{i}. Texto: {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"   Predição: {sentiment}")
        print(f"   Confiança: {confidence*100:.2f}%")


def main():
    """
    Função principal para executar o fine-tuning completo.
    """
    print("="*70)
    print("FINE-TUNING: TRANSFORMER PARA ANÁLISE DE SENTIMENTOS")
    print(f"Modelo: {MODEL_NAME}")
    print("="*70)
    
    # Verificar device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memória disponível: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 1. Carregar dados
    df = load_data('data/sentiment140_processed.csv', sample_size=None)
    
    # 2. Preparar datasets
    train_dataset, eval_dataset = prepare_datasets(df, test_size=0.2)
    
    # 3. Carregar tokenizer
    print("\n" + "="*70)
    print("CARREGANDO TOKENIZER E MODELO")
    print("="*70)
    
    print(f"\nCarregando tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("✓ Tokenizer carregado!")
    
    # 4. Tokenizar datasets
    train_dataset, eval_dataset = tokenize_datasets(
        train_dataset, eval_dataset, tokenizer
    )
    
    # 5. Carregar modelo
    print(f"\nCarregando modelo: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )
    model.to(device)
    print("✓ Modelo carregado!")
    print(f"  - Parâmetros totais: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Parâmetros treináveis: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 6. Criar trainer
    trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset)
    
    # 7. Treinar modelo
    train_result = train_model(trainer)
    
    # 8. Avaliar modelo
    eval_results = evaluate_model(trainer)
    
    # 9. Salvar modelo
    save_model(trainer, tokenizer)
    
    # 10. Testar predições
    test_predictions(model, tokenizer, device)
    
    # Resumo final
    print("\n" + "="*70)
    print("RESUMO FINAL")
    print("="*70)
    print(f"\nModelo: {MODEL_NAME}")
    print(f"Épocas treinadas: {NUM_EPOCHS}")
    print(f"Exemplos de treino: {len(train_dataset):,}")
    print(f"Exemplos de validação: {len(eval_dataset):,}")
    print(f"\nMelhores métricas:")
    print(f"  - Accuracy: {eval_results['eval_accuracy']:.4f} ({eval_results['eval_accuracy']*100:.2f}%)")
    print(f"  - F1-Score: {eval_results['eval_f1']:.4f}")
    print(f"  - Precision: {eval_results['eval_precision']:.4f}")
    print(f"  - Recall: {eval_results['eval_recall']:.4f}")
    print(f"\nModelo salvo em: ./models/sentiment_transformer")
    
    print("\n" + "="*70)
    print("✓ FINE-TUNING CONCLUÍDO COM SUCESSO!")
    print("="*70)


if __name__ == "__main__":
    main()
