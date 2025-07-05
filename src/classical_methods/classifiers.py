import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import pickle

class BaseClassifier:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> None:
        predictions = self.predict(X)
        print(f"\nОценка на {self.model_name}:")
        print(classification_report(y, predictions))

    def save_model(self, filepath: str) -> None:
        if self.model:
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Запазен модел {self.model_name} в {filepath}")

class NaiveBayesClassifier(BaseClassifier):
    def __init__(self):
        super().__init__("Naive Bayes")
        self.model = MultinomialNB()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        param_grid = {'alpha': [0.01, 0.1, 0.5, 1.0]}
        grid = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='f1_weighted')
        grid.fit(X, y)
        self.model = grid.best_estimator_
        print(f"Най-добри параметри за Naive Bayes: {grid.best_params_}")

class LogisticRegressionClassifier(BaseClassifier):
    def __init__(self):
        super().__init__("Logistic Regression")
        self.model = LogisticRegression(max_iter=1000, multi_class='multinomial', class_weight='balanced')

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X)
        param_grid = {'C': [0.01, 0.1, 1, 10]}
        grid = GridSearchCV(LogisticRegression(max_iter=1000, multi_class='multinomial', class_weight='balanced'),
                           param_grid, cv=5, scoring='f1_weighted')
        grid.fit(X_scaled, y)
        self.model = grid.best_estimator_
        print(f"Най-добри параметри за Logistic Regression: {grid.best_params_}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

class SVMClassifier(BaseClassifier):
    def __init__(self):
        super().__init__("SVM")
        self.model = LinearSVC(max_iter=5000, class_weight='balanced')

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X)
        param_grid = {'C': [0.01, 0.1, 1, 10]}
        grid = GridSearchCV(LinearSVC(max_iter=5000, class_weight='balanced'),
                           param_grid, cv=5, scoring='f1_weighted')
        grid.fit(X_scaled, y)
        self.model = grid.best_estimator_
        print(f"Най-добри параметри за SVM: {grid.best_params_}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)