import pandas as pd
import spacy
import re
import numpy as np
import fasttext
from typing import List, Dict, Tuple
from collections import Counter
import torch
from dataset_loader import TextDataLoader


class TextPreprocessorForLSTM:
    """Class for preprocessing Bulgarian texts for LSTM with FastText embeddings."""
    
    def __init__(self, fasttext_model_path: str = None, use_spacy: bool = True, 
                 max_length: int = 200, embedding_dim: int = 300):
        """
        Initializes TextPreprocessorForLSTM.
        
        Args:
            fasttext_model_path (str): Path to pretrained FastText model (e.g., .bin file).
            use_spacy (bool): Whether to use spaCy for POS tagging.
            max_length (int): Maximum sequence length for LSTM input.
            embedding_dim (int): Dimension of FastText embeddings.
        """
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.use_spacy = use_spacy
        self.nlp = None
        if use_spacy:
            try:
                self.nlp = spacy.load('bg_core_news_sm', disable=['ner', 'lemmatizer'])
            except Exception as e:
                print(f"Error loading spaCy model: {e}. Falling back to basic processing.")
                self.use_spacy = False
        
        try:
            self.fasttext_model = fasttext.load_model(fasttext_model_path)
        except Exception as e:
            print(f"Error loading FastText model: {e}. Using dummy embeddings.")
            self.fasttext_model = None
        
        self.vocab = {'<PAD>': 0, '<UNK>': 1}  
        self.word2idx = {}  
        self.idx2word = {} 
        self.embedding_matrix = None  

    def _extract_style_features(self, text: str) -> Dict[str, float]:
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
        
        # POS tag frequencies (if spaCy is available)
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            pos_counts = Counter(token.pos_ for token in doc)
            total_tokens = len(doc)
            for pos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'CONJ', 'PRON', 'PREP']:
                features[f'pos_{pos}'] = pos_counts.get(pos, 0) / total_tokens if total_tokens > 0 else 0
        
        return features

    def build_vocabulary(self, texts: List[str]) -> None:
        """
        Builds vocabulary and embedding matrix from texts.
        
        Args:
            texts (List[str]): List of texts to build vocabulary.
        """
        word_counts = Counter()
        for text in texts:
            words = text.split()  
            word_counts.update(words)
        
  
        idx = len(self.vocab)
        for word, count in word_counts.items():
            if word not in self.vocab and count >= 5:  
                self.vocab[word] = idx
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        # Build embedding matrix
        self.embedding_matrix = np.zeros((len(self.vocab), self.embedding_dim))
        for word, idx in self.vocab.items():
            if word in ['<PAD>', '<UNK>']:
                continue
            if self.fasttext_model:
                self.embedding_matrix[idx] = self.fasttext_model.get_word_vector(word)
            else:
                self.embedding_matrix[idx] = np.random.normal(0, 0.1, self.embedding_dim)

    def preprocess_texts(self, df: pd.DataFrame, text_column: str = 'text', 
                        use_style_features: bool = True) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Preprocesses texts for LSTM with FastText embeddings.
        
        Args:
            df (pd.DataFrame): DataFrame with text and author columns.
            text_column (str): Name of the column containing the text.
            use_style_features (bool): Whether to include style-based features.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[str]]: Sequence tensor, style feature tensor, feature names.
        """
        # Build vocabulary
        if not self.embedding_matrix:
            self.build_vocabulary(df[text_column].tolist())
        
        # Convert texts to sequences
        sequences = []
        for text in df[text_column]:
            words = text.split()[:self.max_length]
            seq = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
            if len(seq) < self.max_length:
                seq += [self.vocab['<PAD>']] * (self.max_length - len(seq))
            sequences.append(seq)
        
        X_sequences = torch.tensor(sequences, dtype=torch.long)
        
        # Extract style features
        style_features = None
        feature_names = []
        if use_style_features:
            style_features = df[text_column].apply(self._extract_style_features)
            style_df = pd.DataFrame(list(style_features))
            X_style = torch.tensor(style_df.to_numpy(), dtype=torch.float)
            feature_names = style_df.columns.tolist()
        else:
            X_style = torch.zeros((len(df), 0), dtype=torch.float)
        
        return X_sequences, X_style, feature_names

    def transform_texts(self, df: pd.DataFrame, text_column: str = 'text', 
                       use_style_features: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transforms new texts for LSTM input.
        
        Args:
            df (pd.DataFrame): DataFrame with text column.
            text_column (str): Name of the column containing the text.
            use_style_features (bool): Whether to include style-based features.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sequence tensor and style feature tensor.
        """
        if not self.embedding_matrix:
            raise ValueError("Vocabulary not built. Call preprocess_texts first.")
        
        # Convert texts to sequences
        sequences = []
        for text in df[text_column]:
            words = text.split()[:self.max_length]
            seq = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
            if len(seq) < self.max_length:
                seq += [self.vocab['<PAD>']] * (self.max_length - len(seq))
            sequences.append(seq)
        
        X_sequences = torch.tensor(sequences, dtype=torch.long)
        
        # Extract style features
        style_features = None
        if use_style_features:
            style_features = df[text_column].apply(self._extract_style_features)
            style_df = pd.DataFrame(list(style_features))
            X_style = torch.tensor(style_df.to_numpy(), dtype=torch.float)
        else:
            X_style = torch.zeros((len(df), 0), dtype=torch.float)
        
        return X_sequences, X_style

def main():
    DATA_DIR = '../data'

    loader = TextDataLoader(DATA_DIR)
    loader.load_train()
    train_data = loader.get_raw_data('train')
    val_data = loader.get_raw_data('val')


    preprocessor = TextPreprocessorForLSTM(
        fasttext_model_path='path_to_fasttext_model.bin',  
        use_spacy=True,
        max_length=200,
        embedding_dim=300
    )
    
    X_train_seq, X_train_style, feature_names = preprocessor.preprocess_texts(
        train_data, text_column='text', use_style_features=True
    )
    y_train = train_data['author']
    
    X_val_seq, X_val_style = preprocessor.transform_texts(
        val_data, text_column='text', use_style_features=True
    )
    y_val = val_data['author']
    
    print("Training sequence tensor shape:", X_train_seq.shape)
    print("Training style tensor shape:", X_train_style.shape)
    print("Feature names:", feature_names)
    print("Validation sequence tensor shape:", X_val_seq.shape)
    print("Validation style tensor shape:", X_val_style.shape)

if __name__ == "__main__":
    main()