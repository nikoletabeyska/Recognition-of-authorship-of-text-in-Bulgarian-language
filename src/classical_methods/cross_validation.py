import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
import csv

def cross_validate_model(classifier, X, y, feature_names, output_dir, n_splits=5):
    print(f"\nКрос-валидация за {classifier.model_name} с {n_splits} гънки...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    f1_scores = []
    top_features_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nОбработка на гънка {fold + 1}...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        classifier.train(X_train, y_train)
        val_predictions = classifier.predict(X_val)
        report = classification_report(y_val, val_predictions, output_dict=True)
        f1 = report['weighted avg']['f1-score']
        f1_scores.append(f1)
        print(f"F1-мерка за гънка {fold + 1}: {f1:.4f}")
        print(classification_report(y_val, val_predictions))

        if isinstance(classifier, (LogisticRegressionClassifier, SVMClassifier)):
            result = permutation_importance(classifier.model, classifier.scaler.transform(X_val), y_val, n_repeats=10, random_state=42)
        else:
            result = permutation_importance(classifier.model, X_val, y_val, n_repeats=10, random_state=42)
        importance = result.importances_mean
        sorted_idx = np.argsort(importance)[::-1]
        top_features = [feature_names[i] for i in sorted_idx[:5]]
        top_features_str = ', '.join([f"{name}: {imp:.4f}" for name, imp in zip(top_features, importance[sorted_idx[:5]])])
        top_features_list.append(top_features_str)

        classifier.save_model(f'{output_dir}/{classifier.model_name.lower().replace(" ", "_")}_fold_{fold + 1}_model.pkl')

    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    print(f"\nСредна F1-мерка: {mean_f1:.4f} (±{std_f1:.4f})")
    return mean_f1, std_f1, top_features_list