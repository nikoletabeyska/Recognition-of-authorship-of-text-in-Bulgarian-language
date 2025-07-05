import pandas as pd
from pathlib import Path
import csv
from data_loader import TextDataLoader
from feature_extractor import FeatureExtractor
from classical_methods.classifiers import NaiveBayesClassifier, LogisticRegressionClassifier, SVMClassifier
from classical_methods.cross_validation import cross_validate_model

def main():
    DATA_DIR = './'
    OUTPUT_DIR = '../output'
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    loader = TextDataLoader(DATA_DIR)
    loader.load_train()
    loader.load_val()
    loader.load_test()

    train_data = loader.get_raw_data('train')
    val_data = loader.get_raw_data('val')
    test_data = loader.get_raw_data('test')
    all_data = pd.concat([train_data, val_data, test_data], ignore_index=True)

    extractor = FeatureExtractor(
        ngram_range_word=(1, 3),
        ngram_range_char=(2, 6),
        max_features_word=2000,
        max_features_char=2000,
        use_stanza_tokenization=True
    )

    features_path = Path(f'{OUTPUT_DIR}/all_features.json')
    word_vectorizer_path = f'{OUTPUT_DIR}/word_tfidf_vectorizer.pkl'
    char_vectorizer_path = f'{OUTPUT_DIR}/char_tfidf_vectorizer.pkl'

    if features_path.exists() and Path(word_vectorizer_path).exists() and Path(char_vectorizer_path).exists():
        print("Зареждане на характеристики от JSON файл...")
        X = extractor.load_features_from_json(str(features_path))
        extractor.load_vectorizers(word_vectorizer_path, char_vectorizer_path)
    else:
        print("Извличане на характеристики...")
        X = extractor.extract_features(all_data['text'].tolist(), 'train')
        extractor.save_features('train', f'{OUTPUT_DIR}/all_features.npy')
        extractor.save_features_to_json('train', f'{OUTPUT_DIR}/all_features.json')
        extractor.save_vectorizers(word_vectorizer_path, char_vectorizer_path)

    y = all_data['author'].values
    feature_names = extractor.get_feature_names()

    results_file = f'{OUTPUT_DIR}/cross_validation_results.csv'
    with open(results_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Classifier', 'Mean F1', 'Std F1', 'Top Features'])

    classifiers = [
        NaiveBayesClassifier(),
        LogisticRegressionClassifier(),
        SVMClassifier()
    ]

    for clf in classifiers:
        mean_f1, std_f1, top_features_list = cross_validate_model(clf, X, y, feature_names, OUTPUT_DIR)
        top_features_str = '; '.join([f"Fold {i+1}: {features}" for i, features in enumerate(top_features_list)])
        with open(results_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([clf.model_name, f"{mean_f1:.4f}", f"{std_f1:.4f}", top_features_str])
        print(f"Резултатите за {clf.model_name} са запазени в {results_file}")

if __name__ == "__main__":
    main()