"""
Script para carregar e processar o dataset Sentiment140 do Hugging Face
"""

import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def load_and_save_data():
    """
    Carrega o dataset Sentiment140 do Hugging Face, processa e salva localmente.
    
    O dataset Sentiment140 contém tweets com sentimentos:
    - 0: Negativo
    - 4: Positivo
    
    Este script mapeia o label 4 para 1, criando um problema binário de classificação.
    """
    print("=" * 70)
    print("CARREGANDO DATASET SENTIMENT140")
    print("=" * 70)
    
    # Carregar o dataset do Hugging Face
    print("\n[1/6] Carregando dataset do Hugging Face...")
    try:
        # O dataset sentiment140 possui split 'train' com 1.6M de exemplos.
        # A partir das versões mais recentes do Hugging Face Datasets, esse dataset
        # requer `trust_remote_code=True` para executar o script remoto.
        dataset = load_dataset("sentiment140", split="train", trust_remote_code=True)
        # Usar apenas caracteres ASCII para evitar problemas de encoding em alguns terminais
        print(f"OK - Dataset carregado com sucesso! Total de exemplos: {len(dataset)}")
    except Exception as e:
        print(f"ERRO ao carregar dataset: {e}")
        return
    
    # Verificar as colunas disponíveis
    print(f"\n[2/6] Colunas disponíveis no dataset: {dataset.column_names}")
    
    # Selecionar apenas as colunas necessárias e converter para DataFrame
    print("\n[3/6] Selecionando colunas 'text' e 'sentiment'...")
    df = pd.DataFrame({
        'text': dataset['text'],
        'sentiment': dataset['sentiment']
    })
    
    print(f"OK - DataFrame criado com shape: {df.shape}")
    print(f"\nDistribuição inicial dos labels:")
    print(df['sentiment'].value_counts().sort_index())
    
    # Filtrar apenas labels 0 e 4 (ignorar label 2 se houver)
    print("\n[4/6] Filtrando apenas labels 0 (negativo) e 4 (positivo)...")
    df = df[df['sentiment'].isin([0, 4])]
    print(f"OK - Shape após filtro: {df.shape}")
    
    # Mapear labels: 4 -> 1 (para problema binário)
    print("\n[5/6] Mapeando labels (4 -> 1)...")
    df['label'] = df['sentiment'].map({0: 0, 4: 1})
    
    # Remover a coluna 'sentiment' original
    df = df.drop(columns=['sentiment'])
    
    # Verificar o mapeamento
    print("OK - Mapeamento concluído!")
    print(f"\nDistribuição final dos labels:")
    print(df['label'].value_counts().sort_index())
    print(f"\n  0 = Negativo: {(df['label'] == 0).sum()} exemplos")
    print(f"  1 = Positivo: {(df['label'] == 1).sum()} exemplos")
    
    # Remover possíveis valores nulos
    df = df.dropna()
    print(f"\nShape final após remover nulos: {df.shape}")
    
    # Salvar o DataFrame processado
    print("\n[6/6] Salvando dados processados...")
    
    # Criar diretório se não existir
    os.makedirs('data', exist_ok=True)
    
    # Salvar como CSV
    output_path = 'data/sentiment140_processed.csv'
    # Usar apenas parâmetros suportados pelo pandas.to_csv
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"OK - Dados salvos com sucesso em: {output_path}")
    
    # Estatísticas finais
    print("\n" + "=" * 70)
    print("RESUMO DO PROCESSAMENTO")
    print("=" * 70)
    print(f"Total de exemplos: {len(df):,}")
    print(f"Número de features: {len(df.columns)}")
    print(f"Colunas: {df.columns.tolist()}")
    print(f"Balanceamento:")
    print(f"  - Classe 0 (Negativo): {(df['label'] == 0).sum():,} ({(df['label'] == 0).mean() * 100:.2f}%)")
    print(f"  - Classe 1 (Positivo): {(df['label'] == 1).sum():,} ({(df['label'] == 1).mean() * 100:.2f}%)")
    print("\nProcessamento concluído!")
    print("=" * 70)
    
    # Mostrar alguns exemplos
    print("\nExemplos do dataset processado:")
    print("\n--- Exemplos NEGATIVOS (label=0) ---")
    for i, row in df[df['label'] == 0].head(3).iterrows():
        print(f"\n{i+1}. {row['text'][:100]}...")
    
    print("\n--- Exemplos POSITIVOS (label=1) ---")
    for i, row in df[df['label'] == 1].head(3).iterrows():
        print(f"\n{i+1}. {row['text'][:100]}...")
    
    return df


def load_processed_data(file_path='data/sentiment140_processed.csv'):
    """
    Carrega o dataset já processado.
    
    Args:
        file_path (str): Caminho do arquivo CSV processado
    
    Returns:
        pd.DataFrame: DataFrame com os dados
    """
    if not os.path.exists(file_path):
        print(f"Arquivo não encontrado: {file_path}")
        print("Execute primeiro a função load_and_save_data() para processar os dados.")
        return None
    
    print(f"Carregando dados de: {file_path}")
    df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
    print(f"✓ Dados carregados! Shape: {df.shape}")
    
    return df


if __name__ == "__main__":
    # Executar o carregamento e processamento dos dados
    df = load_and_save_data()
