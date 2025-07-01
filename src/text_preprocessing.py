import pandas as pd
import stanza
import re
from typing import List, Dict, Tuple
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy as np
from dataset_loader import TextDataLoader

class TextPreprocessor:
    """Class for preprocessing Bulgarian texts for authorship attribution."""
    
    def __init__(self, use_stanza: bool = True, stop_words: List[str] = None):
        """
        Initializes TextPreprocessor for Bulgarian texts.
        
        Args:
            use_stanza (bool): Whether to use Stanza for POS tagging (requires 'bg' model).
            stop_words (List[str]): Optional list of stop words for Bulgarian (if None, no stop words are removed).
        """
        self.use_stanza = use_stanza
        self.stop_words = stop_words if stop_words else []
        self.nlp = None
        if use_stanza:
            try:
                self.nlp = stanza.Pipeline('bg', processors='tokenize,pos', use_gpu=False, download_method=stanza.DownloadMethod.REUSE_RESOURCES)
                print("Stanza model for Bulgarian loaded successfully.")
            except Exception as e:
                print(f"Error loading Stanza model: {e}. Falling back to basic processing.")
                self.use_stanza = False
        self.vectorizer = None
        self.feature_names = None

    def extract_style_features(self, text: str) -> Dict[str, float]:
        """
        Extracts style-based features from a text.
        
        Args:
            text (str): Input text.
            
        Returns:
            Dict[str, float]: Dictionary with style features.
        """
        features = {}
        
        # Sentence length features
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_lengths = [len(s.split()) for s in sentences]
        features['avg_sentence_length'] = np.mean(sentence_lengths) if sentence_lengths else 0
        features['std_sentence_length'] = np.std(sentence_lengths) if sentence_lengths else 0
        
        # Punctuation frequency
        punctuation_counts = Counter(re.findall(r'[.,!?;:"\'()-]', text))
        for punct in [',', '.', '!', '?', ';', ':', '"', '\'', '(', ')', '-']:
            features[f'punct_{punct}'] = punctuation_counts.get(punct, 0) / len(text)
        
        # Word length features
        words = text.split()
        word_lengths = [len(word) for word in words]
        features['avg_word_length'] = np.mean(word_lengths) if word_lengths else 0
        
        # POS tag frequencies
        if self.use_stanza and self.nlp:
            try:
                doc = self.nlp(text)
                pos_counts = Counter(word.pos for sent in doc.sentences for word in sent.words)
                total_tokens = sum(pos_counts.values())
                for pos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'CONJ', 'PRON', 'PREP']:
                    features[f'pos_{pos}'] = pos_counts.get(pos, 0) / total_tokens if total_tokens > 0 else 0
            except Exception as e:
                print(f"Error processing text with Stanza: {e}. Skipping POS features.")
                for pos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'CONJ', 'PRON', 'PREP']:
                    features[f'pos_{pos}'] = 0
        
        return features

    def preprocess_texts(self, df: pd.DataFrame, text_column: str = 'text', 
                        use_tfidf: bool = True, use_ngrams: bool = True, 
                        use_style_features: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Preprocesses texts and extracts features for classification.
        
        Args:
            df (pd.DataFrame): DataFrame with text and author columns.
            text_column (str): Name of the column containing the text.
            use_tfidf (bool): Whether to include TF-IDF features.
            use_ngrams (bool): Whether to include n-gram features.
            use_style_features (bool): Whether to include style-based features.
            
        Returns:
            Tuple[np.ndarray, List[str]]: Feature matrix and feature names.
        """
        if df.empty or text_column not in df.columns:
            print(f"Error: DataFrame is empty or '{text_column}' column is missing.")
            return np.array([]), []
        
        X_features = []
        feature_names = []
        
        # TF-IDF features
        if use_tfidf:
            if not self.vectorizer:
                self.vectorizer = TfidfVectorizer(
                    max_features=5000, 
                    lowercase=False,  
                    token_pattern=r'\S+',  
                    ngram_range=(1, 2) if use_ngrams else (1, 1)
                )
            X_tfidf = self.vectorizer.fit_transform(df[text_column])
            X_features.append(X_tfidf)
            feature_names.extend(self.vectorizer.get_feature_names_out())
        else:
            self.vectorizer = None
        
        # Style-based features
        if use_style_features:
            style_features = df[text_column].apply(self.extract_style_features)
            style_df = pd.DataFrame(list(style_features))
            X_style = style_df.to_numpy()
            X_features.append(X_style)
            feature_names.extend(style_df.columns)
        
        # Combine features
        if len(X_features) > 1:
            X_combined = hstack(X_features)
        else:
            X_combined = X_features[0]
        
        self.feature_names = feature_names
        return X_combined, feature_names

    def transform_texts(self, df: pd.DataFrame, text_column: str = 'text') -> np.ndarray:
        """
        Transforms new texts using the fitted vectorizer and style features.
        
        Args:
            df (pd.DataFrame): DataFrame with text column.
            text_column (str): Name of the column containing the text.
            
        Returns:
            np.ndarray: Transformed feature matrix.
        """
        if df.empty or text_column not in df.columns:
            print(f"Error: DataFrame is empty or '{text_column}' column is missing.")
            return np.array([])
        
        X_features = []
        
        if self.vectorizer:
            X_tfidf = self.vectorizer.transform(df[text_column])
            X_features.append(X_tfidf)
        
        style_features = df[text_column].apply(self.extract_style_features)
        style_df = pd.DataFrame(list(style_features))
        X_style = style_df.to_numpy()
        X_features.append(X_style)
        
        if len(X_features) > 1:
            return hstack(X_features)
        return X_features[0]

    def get_feature_names(self) -> List[str]:
        """Returns the names of the extracted features."""
        return self.feature_names

def main():
    DATA_DIR = '../data'

    loader = TextDataLoader(DATA_DIR)
    loader.load_train()
    train_data = loader.get_raw_data('train')
    
    if train_data.empty:
        print("Error: Failed to load training data. Please ensure train_data.json exists and contains valid data.")
        return
    
    print("Train data columns:", train_data.columns)
    print("Train data sample:", train_data.head())
    
    preprocessor = TextPreprocessor(use_stanza=True)
    
    X_train, feature_names = preprocessor.preprocess_texts(
        train_data, text_column='text', use_tfidf=True, use_ngrams=True, use_style_features=True
    )
    y_train = train_data['author']
    
    print("Feature matrix shape:", X_train.shape)
    print("Feature names (first 10):", feature_names[:50])
    print("Author classes:", y_train.unique())

if __name__ == "__main__":
    main()