"""
Funções de limpeza e preparação de texto para análise de sentimentos
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download de recursos necessários do NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Baixando 'punkt'...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Baixando 'stopwords'...")
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Baixando 'wordnet'...")
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    print("Baixando 'omw-1.4'...")
    nltk.download('omw-1.4')


# Inicializar stopwords e lematizador
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text):
    """
    Limpa e preprocessa um texto para análise de sentimentos.
    
    Passos de limpeza:
    1. Converte para minúsculas
    2. Remove URLs (http/https)
    3. Remove menções (@usuário) e hashtags (#hashtag)
    4. Remove pontuação e números
    5. Remove stopwords em inglês
    6. Aplica lematização
    7. Remove espaços extras
    
    Args:
        text (str): Texto a ser limpo
    
    Returns:
        str: Texto limpo e preprocessado
    """
    # Verificar se o texto é válido
    if not isinstance(text, str):
        return ""
    
    # 1. Converter para minúsculas
    text = text.lower()
    
    # 2. Remover URLs (http/https)
    text = re.sub(r'http\S+|https\S+|www\.\S+', '', text)
    
    # 3. Remover menções (@usuário) e hashtags (#hashtag)
    text = re.sub(r'@\w+', '', text)  # Remove menções
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    
    # 4. Remover pontuação e números (manter apenas letras e espaços)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remover espaços extras antes da tokenização
    text = ' '.join(text.split())
    
    # 5. Tokenizar o texto
    tokens = word_tokenize(text)
    
    # 6. Remover stopwords
    tokens = [token for token in tokens if token not in STOP_WORDS and len(token) > 1]
    
    # 7. Aplicar lematização
    tokens = [LEMMATIZER.lemmatize(token) for token in tokens]
    
    # Juntar os tokens de volta em uma string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text


def clean_text_batch(texts, show_progress=True):
    """
    Aplica limpeza em um batch de textos.
    
    Args:
        texts (list or pd.Series): Lista de textos a serem limpos
        show_progress (bool): Mostrar barra de progresso
    
    Returns:
        list: Lista de textos limpos
    """
    if show_progress:
        from tqdm import tqdm
        return [clean_text(text) for text in tqdm(texts, desc="Limpando textos")]
    else:
        return [clean_text(text) for text in texts]


if __name__ == "__main__":
    # Testes da função clean_text
    print("="*70)
    print("TESTE DA FUNÇÃO clean_text()")
    print("="*70)
    
    # Exemplos de teste
    test_texts = [
        "Check out this amazing product! https://example.com @user #awesome",
        "I HATE this movie!!! It's terrible... 😢",
        "Best day ever!!! Love it so much ❤️ #happy #blessed",
        "Call me at 555-1234 or visit http://test.com for more info @john",
        "This is a normal sentence with some stopwords like the, is, a, an."
    ]
    
    print("\nTestes:")
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Original:")
        print(f"   {text}")
        print(f"   Limpo:")
        print(f"   {clean_text(text)}")
    
    print("\n" + "="*70)
    print("COMPARAÇÃO: ANTES E DEPOIS")
    print("="*70)
    
    sample = "RT @user: I really LOVE this product!!! 😍 Check it out: https://example.com #amazing #musthave"
    print(f"\nTexto original ({len(sample)} caracteres):")
    print(f"'{sample}'")
    
    cleaned = clean_text(sample)
    print(f"\nTexto limpo ({len(cleaned)} caracteres):")
    print(f"'{cleaned}'")
    
    print("\n✓ Função clean_text() testada com sucesso!")
