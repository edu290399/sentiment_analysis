"""
Script para fazer predições usando modelos treinados
Suporta: Modelo Baseline e Transformer
"""

import os
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Importar função de preprocessamento
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar do arquivo 2_preprocessing.py
import importlib.util
spec = importlib.util.spec_from_file_location("preprocessing", 
                                               os.path.join(os.path.dirname(__file__), "2_preprocessing.py"))
preprocessing_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocessing_module)
clean_text = preprocessing_module.clean_text


class SentimentPredictor:
    """Classe para fazer predições de sentimento"""
    
    def __init__(self, model_type='baseline'):
        """
        Inicializa o preditor.
        
        Args:
            model_type (str): Tipo de modelo ('baseline' ou 'transformer')
        """
        self.model_type = model_type
        
        if model_type == 'baseline':
            self.load_baseline_model()
        elif model_type == 'transformer':
            self.load_transformer_model()
        else:
            raise ValueError("model_type deve ser 'baseline' ou 'transformer'")
    
    def load_baseline_model(self):
        """Carrega modelo baseline (TF-IDF + Logistic Regression)"""
        model_path = 'models/baseline_model.pkl'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo baseline não encontrado em: {model_path}")
        
        print(f"Carregando modelo baseline de: {model_path}")
        self.pipeline = joblib.load(model_path)
        print("✓ Modelo baseline carregado!")
    
    def load_transformer_model(self):
        """Carrega modelo Transformer (DistilBERT)"""
        model_path = 'models/sentiment_transformer'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo transformer não encontrado em: {model_path}")
        
        print(f"Carregando modelo transformer de: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Modelo transformer carregado! (Device: {self.device})")
    
    def predict_baseline(self, texts):
        """
        Faz predições usando modelo baseline.
        
        Args:
            texts (list): Lista de textos
        
        Returns:
            tuple: (predictions, probabilities)
        """
        # Limpar textos
        cleaned_texts = [clean_text(text) for text in texts]
        
        # Predições
        predictions = self.pipeline.predict(cleaned_texts)
        probabilities = self.pipeline.predict_proba(cleaned_texts)
        
        return predictions, probabilities
    
    def predict_transformer(self, texts, max_length=128):
        """
        Faz predições usando modelo Transformer.
        
        Args:
            texts (list): Lista de textos
            max_length (int): Comprimento máximo dos tokens
        
        Returns:
            tuple: (predictions, probabilities)
        """
        predictions = []
        probabilities = []
        
        for text in texts:
            # Tokenizar
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding='max_length'
            ).to(self.device)
            
            # Predição
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                pred = probs.argmax()
            
            predictions.append(pred)
            probabilities.append(probs)
        
        return predictions, probabilities
    
    def predict(self, texts):
        """
        Faz predições (detecta automaticamente o tipo de modelo).
        
        Args:
            texts (str or list): Texto ou lista de textos
        
        Returns:
            tuple: (predictions, probabilities)
        """
        # Converter para lista se for string única
        if isinstance(texts, str):
            texts = [texts]
        
        if self.model_type == 'baseline':
            return self.predict_baseline(texts)
        else:
            return self.predict_transformer(texts)
    
    def predict_with_labels(self, texts):
        """
        Faz predições e retorna com labels descritivos.
        
        Args:
            texts (str or list): Texto ou lista de textos
        
        Returns:
            list: Lista de dicionários com predições
        """
        predictions, probabilities = self.predict(texts)
        
        results = []
        for i, (text, pred, prob) in enumerate(zip(texts, predictions, probabilities)):
            sentiment = "Positivo" if pred == 1 else "Negativo"
            confidence = prob[pred] * 100
            
            results.append({
                'text': text,
                'sentiment': sentiment,
                'label': int(pred),
                'confidence': confidence,
                'prob_negative': prob[0] * 100,
                'prob_positive': prob[1] * 100
            })
        
        return results


def compare_models(texts):
    """
    Compara predições de ambos os modelos (baseline e transformer).
    
    Args:
        texts (list): Lista de textos para comparar
    """
    print("="*80)
    print("COMPARAÇÃO: BASELINE vs TRANSFORMER")
    print("="*80)
    
    # Carregar modelos
    try:
        print("\nCarregando modelos...")
        baseline = SentimentPredictor('baseline')
        transformer = SentimentPredictor('transformer')
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        print("Certifique-se de que ambos os modelos foram treinados primeiro.")
        return
    
    # Fazer predições
    print("\nFazendo predições...\n")
    baseline_results = baseline.predict_with_labels(texts)
    transformer_results = transformer.predict_with_labels(texts)
    
    # Mostrar comparação
    for i, text in enumerate(texts):
        b_res = baseline_results[i]
        t_res = transformer_results[i]
        
        print(f"\n{i+1}. Texto: {text[:70]}{'...' if len(text) > 70 else ''}")
        print(f"   {'='*76}")
        print(f"   BASELINE:    {b_res['sentiment']:8s} (confiança: {b_res['confidence']:5.2f}%)")
        print(f"   TRANSFORMER: {t_res['sentiment']:8s} (confiança: {t_res['confidence']:5.2f}%)")
        
        # Indicar se houve concordância
        if b_res['label'] == t_res['label']:
            print(f"   ✓ CONCORDAM")
        else:
            print(f"   ✗ DISCORDAM")


def main():
    """Função principal para demonstrar uso do preditor"""
    print("="*80)
    print("SISTEMA DE PREDIÇÃO DE SENTIMENTOS")
    print("="*80)
    
    # Textos de exemplo
    test_texts = [
        "I absolutely love this product! It's amazing and works perfectly!",
        "This is the worst experience I've ever had. Terrible service.",
        "Great quality and fast delivery. Highly recommend!",
        "Disappointed with the purchase. Not worth the money.",
        "It's okay, nothing special but does the job.",
        "Best decision ever! So happy with this purchase!",
        "Waste of time and money. Very disappointed.",
        "Pretty good overall. Met my expectations.",
        "Fantastic! Exceeded all my expectations!",
        "Poor quality, broke after one day. Avoid!"
    ]
    
    # Escolher modelo
    print("\nEscolha o modelo:")
    print("1. Baseline (TF-IDF + Logistic Regression)")
    print("2. Transformer (DistilBERT)")
    print("3. Comparar ambos")
    
    choice = input("\nOpção (1/2/3): ").strip()
    
    if choice == '3':
        # Comparar modelos
        compare_models(test_texts)
    else:
        # Usar modelo individual
        model_type = 'baseline' if choice == '1' else 'transformer'
        
        try:
            predictor = SentimentPredictor(model_type)
            
            print(f"\n{'='*80}")
            print(f"PREDIÇÕES - Modelo: {model_type.upper()}")
            print(f"{'='*80}\n")
            
            results = predictor.predict_with_labels(test_texts)
            
            for i, res in enumerate(results, 1):
                emoji = "✅" if res['sentiment'] == "Positivo" else "❌"
                print(f"{i}. Texto: {res['text'][:65]}{'...' if len(res['text']) > 65 else ''}")
                print(f"   Sentimento: {res['sentiment']} {emoji}")
                print(f"   Confiança: {res['confidence']:.2f}%")
                print(f"   Probabilidades: Neg={res['prob_negative']:.2f}% | Pos={res['prob_positive']:.2f}%")
                print()
        
        except FileNotFoundError as e:
            print(f"\nErro: {e}")
            print("Treine o modelo primeiro usando:")
            if model_type == 'baseline':
                print("  python src/3_train_baseline.py")
            else:
                print("  python src/4_train_transformer.py")


if __name__ == "__main__":
    main()

