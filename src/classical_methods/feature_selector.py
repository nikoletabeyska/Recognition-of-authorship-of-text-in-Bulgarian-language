import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re
from typing import List
import json
import stanza

nlp = None

def get_nlp_pipeline():
    global nlp
    if nlp is None:
        print("Изтегляне и инициализиране на Stanza за българска токенизация...")
        try:
            stanza.download('bg')
            nlp = stanza.Pipeline('bg', processors='tokenize')
            print("Stanza токенизаторът е инициализиран успешно.")
        except Exception as e:
            print(f"Грешка при инициализиране на Stanza: {e}")
            nlp = None
    return nlp

class FeatureExtractor:
    def __init__(self, ngram_range_word=(1, 3), ngram_range_char=(2, 6),
                 max_features_word=2000, max_features_char=2000, use_stanza_tokenization=False):
        self.word_vectorizer = TfidfVectorizer(
            analyzer='word', ngram_range=ngram_range_word,
            max_features=max_features_word, lowercase=True
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer='char', ngram_range=ngram_range_char,
            max_features=max_features_char, lowercase=True
        )
        self.function_words = [
            'и', 'или', 'но', 'защото', 'ако', 'че', 'в', 'на', 'с', 'за',
            'от', 'до', 'при', 'по', 'над', 'под', 'между', 'през', 'след',
            'преди', 'обаче', 'все пак', 'понеже', 'дори', 'като', 'щом',
            'докато', 'за да', 'освен', 'без', 'се', 'той', 'тя', 'те', 'аз', 'ти'
        ]
        self.feature_cache = {}
        self.use_stanza_tokenization = use_stanza_tokenization

    def extract_features(self, texts: List[str], split: str) -> np.ndarray:
        print(f"Извличане на характеристики за {split} набор с {len(texts)} текста...")
        if split == 'train':
            word_ngram_features = self.word_vectorizer.fit_transform(texts).toarray()
        else:
            word_ngram_features = self.word_vectorizer.transform(texts).toarray()
        print(f"N-грами(On думи: {word_ngram_features.shape}")
        if split == 'train':
            char_ngram_features = self.char_vectorizer.fit_transform(texts).toarray()
        else:
            char_ngram_features = self.char_vectorizer.transform(texts).toarray()
        print(f"N-грами на символи: {char_ngram_features.shape}")
        stylistic_features = [self._extract_basic_features(text) for text in texts]
        print(f"Стилистични характеристики: {len(stylistic_features)} x {len(stylistic_features[0])}")
        features = np.hstack([word_ngram_features, char_ngram_features, np.array(stylistic_features)])
        print(f"Форма на комбинираните характеристики: {features.shape}")
        self.feature_cache[split] = features
        return features

    def _extract_basic_features(self, text: str) -> List[float]:
        if self.use_stanza_tokenization:
            nlp_pipeline = get_nlp_pipeline()
            if nlp_pipeline:
                doc = nlp_pipeline(text[:2000])
                sentences = [sent.text for sent in doc.sentences]
                sentences = [s.strip() for s in sentences if s.strip()]
                words = [word.text.lower() for sent in doc.sentences for word in sent.words]
            else:
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                words = re.findall(r'\b\w+\b', text.lower())
        else:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            words = re.findall(r'\b\w+\b', text.lower())
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        unique_words = len(set(words))
        ttr = unique_words / len(words) if words else 0
        func_word_count = sum(1 for word in words if word in self.function_words)
        func_word_freq = func_word_count / len(words) if words else 0
        punctuation_chars = len(re.findall(r'[^\w\s]', text))
        punctuation = punctuation_chars / len(text) if text else 0
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        syllable_count = sum(sum(1 for char in word if char in 'аеиоуя') for word in words) / len(words) if words else 0
        return [avg_sentence_length, ttr, func_word_freq, punctuation, avg_word_length, syllable_count]

    def save_features(self, split: str, filepath: str) -> None:
        if split in self.feature_cache:
            np.save(filepath, self.feature_cache[split])
            print(f"Запазени {split} характеристики в {filepath}")
        else:
            print(f"Няма характеристики за {split} набор")

    def save_features_to_json(self, split: str, filepath: str, feature_names: List[str] = None) -> None:
        if split not in self.feature_cache:
            print(f"Няма характеристики за {split} набор")
            return
        if feature_names is None:
            feature_names = self.get_feature_names()
        features = self.feature_cache[split]
        feature_dicts = [
            {name: float(value) for name, value in zip(feature_names, feature_row)}
            for feature_row in features
        ]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(feature_dicts, f, ensure_ascii=False, indent=2)
        print(f"Запазени {split} характеристики в {filepath} като JSON")

    def load_features_from_json(self, filepath: str) -> np.ndarray:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                feature_dicts = json.load(f)
            feature_names = list(feature_dicts[0].keys())
            features = np.array([[d[name] for name in feature_names] for d in feature_dicts])
            print(f"Заредени характеристики от {filepath}: {features.shape}")
            return features
        except Exception as e:
            print(f"Грешка при зареждане на характеристики от {filepath}: {e}")
            return np.array([])

    def save_vectorizers(self, word_filepath: str, char_filepath: str) -> None:
        with open(word_filepath, 'wb') as f:
            pickle.dump(self.word_vectorizer, f)
        with open(char_filepath, 'wb') as f:
            pickle.dump(self.char_vectorizer, f)
        print(f"Запазен векторизатор за думи в {word_filepath}")
        print(f"Запазен векторизатор за символи в {char_filepath}")

    def load_vectorizers(self, word_filepath: str, char_filepath: str) -> None:
        with open(word_filepath, 'rb') as f:
            self.word_vectorizer = pickle.load(f)
        with open(char_filepath, 'rb') as f:
            self.char_vectorizer = pickle.load(f)
        print(f"Зареден векторизатор за думи от {word_filepath}")
        print(f"Зареден векторизатор за символи от {char_filepath}")

    def get_feature_names(self) -> List[str]:
        return (list(self.word_vectorizer.get_feature_names_out()) +
                list(self.char_vectorizer.get_feature_names_out()) +
                ['avg_sentence_length', 'ttr', 'func_word_freq', 'punctuation', 'avg_word_length', 'syllable_count'])