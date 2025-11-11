"""
Funções utilitárias para o projeto de análise de sentimentos
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


def save_model(model, path, filename):
    """
    Salva um modelo usando joblib
    
    Args:
        model: Modelo a ser salvo
        path (str): Diretório para salvar
        filename (str): Nome do arquivo
    """
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    joblib.dump(model, full_path)
    print(f"Modelo salvo em: {full_path}")


def load_model(path, filename):
    """
    Carrega um modelo usando joblib
    
    Args:
        path (str): Diretório do modelo
        filename (str): Nome do arquivo
    
    Returns:
        Modelo carregado
    """
    full_path = os.path.join(path, filename)
    model = joblib.load(full_path)
    print(f"Modelo carregado de: {full_path}")
    return model


def calculate_metrics(y_true, y_pred, average='weighted'):
    """
    Calcula métricas de classificação
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições
        average (str): Tipo de média para métricas
    
    Returns:
        dict: Dicionário com métricas
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average),
        'recall': recall_score(y_true, y_pred, average=average),
        'f1': f1_score(y_true, y_pred, average=average)
    }
    
    return metrics


def print_metrics(y_true, y_pred):
    """
    Imprime métricas de classificação
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições
    """
    metrics = calculate_metrics(y_true, y_pred)
    
    print("=== Métricas de Classificação ===")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred))


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    """
    Plota matriz de confusão
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Predições
        labels (list): Labels das classes
        save_path (str): Caminho para salvar a figura
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Matriz de Confusão')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Predito')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figura salva em: {save_path}")
    
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """
    Plota curva ROC
    
    Args:
        y_true: Labels verdadeiros
        y_pred_proba: Probabilidades preditas
        save_path (str): Caminho para salvar a figura
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figura salva em: {save_path}")
    
    plt.show()


def save_metrics_to_json(metrics, path, filename='metrics.json'):
    """
    Salva métricas em arquivo JSON
    
    Args:
        metrics (dict): Dicionário com métricas
        path (str): Diretório para salvar
        filename (str): Nome do arquivo
    """
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    
    with open(full_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Métricas salvas em: {full_path}")


def load_metrics_from_json(path, filename='metrics.json'):
    """
    Carrega métricas de arquivo JSON
    
    Args:
        path (str): Diretório do arquivo
        filename (str): Nome do arquivo
    
    Returns:
        dict: Dicionário com métricas
    """
    full_path = os.path.join(path, filename)
    
    with open(full_path, 'r') as f:
        metrics = json.load(f)
    
    print(f"Métricas carregadas de: {full_path}")
    return metrics


def compare_models(results_dict):
    """
    Compara resultados de múltiplos modelos
    
    Args:
        results_dict (dict): Dicionário com resultados {modelo: métricas}
    
    Returns:
        pd.DataFrame: DataFrame com comparação
    """
    df = pd.DataFrame(results_dict).T
    df = df.sort_values('f1', ascending=False)
    
    print("=== Comparação de Modelos ===")
    print(df)
    
    return df


def plot_model_comparison(results_dict, save_path=None):
    """
    Plota comparação visual de modelos
    
    Args:
        results_dict (dict): Dicionário com resultados {modelo: métricas}
        save_path (str): Caminho para salvar a figura
    """
    df = pd.DataFrame(results_dict).T
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        df[metric].plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title(f'{metric.capitalize()} por Modelo')
        ax.set_ylabel(metric.capitalize())
        ax.set_xlabel('Modelo')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figura salva em: {save_path}")
    
    plt.show()


def create_directory_structure(base_path='.'):
    """
    Cria estrutura de diretórios do projeto
    
    Args:
        base_path (str): Caminho base do projeto
    """
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'notebooks',
        'src',
        'results/figures',
        'results/reports'
    ]
    
    for directory in directories:
        full_path = os.path.join(base_path, directory)
        os.makedirs(full_path, exist_ok=True)
        print(f"Diretório criado: {full_path}")


if __name__ == "__main__":
    # Exemplo de uso
    print("Testando funções utilitárias...")
    
    # Criar estrutura de diretórios
    create_directory_structure()
    
    # Exemplo de métricas
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0])
    
    print("\n")
    print_metrics(y_true, y_pred)

