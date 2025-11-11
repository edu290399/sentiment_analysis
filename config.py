"""
Configurações globais do projeto de Análise de Sentimentos
"""

import os

# ============================================================================
# CAMINHOS
# ============================================================================

# Diretórios principais
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')

# Arquivos de dados
SENTIMENT140_PROCESSED = os.path.join(DATA_DIR, 'sentiment140_processed.csv')

# Modelos salvos
BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, 'baseline_model.pkl')
TRANSFORMER_MODEL_PATH = os.path.join(MODELS_DIR, 'sentiment_transformer')

# ============================================================================
# CONFIGURAÇÕES DE DADOS
# ============================================================================

# Dataset
DATASET_NAME = 'sentiment140'
TEXT_COLUMN = 'text'
LABEL_COLUMN = 'label'

# Labels
LABEL_MAP = {
    0: 'Negativo',
    1: 'Positivo'
}

# Split de dados
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================================
# CONFIGURAÇÕES DE PREPROCESSAMENTO
# ============================================================================

# NLTK
STOPWORDS_LANGUAGE = 'english'

# Limpeza de texto
REMOVE_HTML = True
REMOVE_URLS = True
REMOVE_MENTIONS = True
REMOVE_HASHTAGS = True
REMOVE_PUNCTUATION = True
REMOVE_NUMBERS = True
TO_LOWERCASE = True
REMOVE_STOPWORDS = True
APPLY_LEMMATIZATION = True

# ============================================================================
# CONFIGURAÇÕES DO MODELO BASELINE
# ============================================================================

# TF-IDF Vectorizer
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)  # Unigramas e bigramas
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_SUBLINEAR_TF = True

# Logistic Regression
LR_MAX_ITER = 1000
LR_C = 1.0
LR_SOLVER = 'lbfgs'
LR_CLASS_WEIGHT = 'balanced'

# ============================================================================
# CONFIGURAÇÕES DO MODELO TRANSFORMER
# ============================================================================

# Modelo
TRANSFORMER_MODEL_NAME = 'distilbert-base-uncased'
NUM_LABELS = 2

# Tokenização
MAX_LENGTH = 128
PADDING = 'max_length'
TRUNCATION = True

# Treinamento
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 2e-5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01

# Estratégias
EVALUATION_STRATEGY = 'epoch'
SAVE_STRATEGY = 'epoch'
METRIC_FOR_BEST_MODEL = 'f1'
SAVE_TOTAL_LIMIT = 2

# ============================================================================
# CONFIGURAÇÕES DE HARDWARE
# ============================================================================

# GPU
USE_GPU = True
FP16 = True  # Mixed precision (apenas se GPU disponível)

# Número de workers para dataloader
NUM_WORKERS = 4

# ============================================================================
# CONFIGURAÇÕES DE LOGGING
# ============================================================================

LOGGING_STEPS = 100
LOG_LEVEL = 'INFO'

# ============================================================================
# OUTRAS CONFIGURAÇÕES
# ============================================================================

# Seed para reprodutibilidade
SEED = 42

# Verbose
VERBOSE = True

# Salvar resultados
SAVE_RESULTS = True
SAVE_PLOTS = True

